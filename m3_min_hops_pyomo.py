import sys
import os
import csv
import time
import numpy as np
import pandas as pd
from gurobipy import Model, GRB, quicksum

# Ajustes de caminho e configuração geral
sys.path.append('modules')
np.set_printoptions(suppress=True)

# Configurações gerais
TOPOLOGY = 'melbourne'
PATH = 'melbourne2/'
COLUMNS = ['id', 'cpu', 'memory', 'storage', 'bandwidth']
TIME_LIMIT = 30  # Limite de tempo em minutos
CAPACITY_PERCENT = 1.0  # Percentual de capacidade (1.0 = 100%)

# Importação de módulos personalizados
import get_data as gd  # type: ignore


def get_data_nodes(topology, capacity_percent):
    """
    Carrega os dados dos nós e ajusta para a topologia.

    Args:
    - topology (str): Nome da topologia.
    - capacity_percent (float): Percentual de capacidade dos nós.

    Returns:
    - nodes (DataFrame): Informações dos nós ajustadas.
    - edges (DataFrame): Informações das conexões entre nós.
    - s_matrix (np.array): Matriz de distâncias entre nós.
    """
    df_nodes, df_edges = gd.get_topology(topology, capacity_percent=capacity_percent)

    if topology == 'melbourne':
        df_nodes = df_nodes.iloc[1:].reset_index(drop=True)  # Remove o primeiro nó (Base Station)
        # Para topologia "melbourne", todos os hops são fixados como 2
        num_nodes = len(df_nodes)
        s_matrix = np.full((num_nodes, num_nodes), 2, dtype=int)
        np.fill_diagonal(s_matrix, 0)  # Distância para si mesmo é 0
    else:
        # Calcula matriz de hops normalmente
        G = gd.create_graph_topology(df_nodes, df_edges)
        s_matrix = gd.floyd_warshall(G, weight='hops')

    return df_nodes[COLUMNS], df_edges, s_matrix


def get_data_application(app_name, path):
    """
    Carrega os dados de uma aplicação.

    Args:
    - app_name (str): Nome da aplicação.
    - path (str): Caminho para os dados da aplicação.

    Returns:
    - services (DataFrame): Informações sobre os serviços.
    - choreography (DataFrame): Informações sobre coreografias.
    """
    df_services, df_choreog = gd.get_application(app_name, path)
    df_services = df_services[COLUMNS]
    df_services.loc[:, 'id'] = range(len(df_services))  # Reindexar serviços
    return df_services, df_choreog


def solve_placement(services, nodes, s_matrix, time_limit):
    """
    Resolve o problema de alocação de serviços em nós com Gurobi.

    Args:
    - services (DataFrame): Informações sobre os serviços.
    - nodes (DataFrame): Informações sobre os nós.
    - s_matrix (np.array): Matriz de distâncias entre nós.
    - time_limit (int): Limite de tempo para o solver (em minutos).

    Returns:
    - allocations (list): Alocações de serviços em nós.
    - nodes_allocated (list): Lista de nós alocados.
    - time_placement (str): Tempo total de execução.
    - solver_time (float): Tempo de execução do solver.
    - status (str): Status da solução.
    """
    num_services = len(services)
    num_nodes = len(nodes)
    resources = ['cpu', 'memory', 'storage', 'bandwidth']

    # Iniciar o modelo Gurobi
    model = Model("ServicePlacement")
    model.setParam("TimeLimit", time_limit * 60)

    # Variáveis de decisão
    x = model.addVars(num_services, num_nodes, vtype=GRB.BINARY, name="x")  # Alocação de serviços
    u = model.addVars(range(1, num_services), num_nodes, num_nodes, vtype=GRB.BINARY, name="u")  # Hops

    # Função objetivo: Minimizar hops
    model.setObjective(
        quicksum(s_matrix[j1, j2] * u[i, j1, j2]
                 for i in range(1, num_services)
                 for j1 in range(num_nodes)
                 for j2 in range(num_nodes)),
        GRB.MINIMIZE
    )

    # Restrições de capacidade dos nós
    for j in range(num_nodes):
        for res in resources:
            model.addConstr(
                quicksum(x[i, j] * services.loc[i, res] for i in range(num_services)) <= nodes.loc[j, res],
                name=f"capacity_{res}_node_{j}"
            )

    # Cada serviço deve ser alocado a exatamente um nó
    for i in range(num_services):
        model.addConstr(
            quicksum(x[i, j] for j in range(num_nodes)) == 1,
            name=f"one_node_service_{i}"
        )

    # Restrições de hops
    for i in range(1, num_services):
        for j1 in range(num_nodes):
            for j2 in range(num_nodes):
                model.addConstr(
                    x[i-1, j1] + x[i, j2] <= u[i, j1, j2] + 1,
                    name=f"hop_{i}_{j1}_{j2}"
                )

    # Resolver o modelo
    start_time = time.time()
    model.optimize()
    time_placement = '{:.8f}'.format(time.time() - start_time)

    # Processar resultados
    allocations = []
    nodes_allocated = []
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        for i in range(num_services):
            for j in range(num_nodes):
                if x[i, j].X > 0.5:
                    allocations.append((i, j))
                    nodes_allocated.append(j)

        status = "Optimal" if model.Status == GRB.OPTIMAL else "Time Limit"
        print(f"Objective (min hops): {model.ObjVal}")
    else:
        status = "Infeasible"
        print("No feasible solution found.")

    return allocations, nodes_allocated, time_placement, model.Runtime, status


def main():
    # Carregar dados da topologia
    nodes, edges, s_matrix = get_data_nodes(TOPOLOGY, CAPACITY_PERCENT)

    # Processar alocação para cada aplicação
    dict_allocations = {}
    for i in range(20):
        app_name = f"App{i}"
        print(f"Processing {app_name}...")
        services, choreography = get_data_application(app_name, PATH)
        allocations, nodes_allocated, time_placement, solver_time, status = solve_placement(
            services, nodes, s_matrix, TIME_LIMIT
        )
        dict_allocations[app_name] = (allocations, time_placement, solver_time, status)

    # Salvar resultados em CSV
    os.makedirs('results', exist_ok=True)
    output_file = f'results/m3_min_hops_{TOPOLOGY}_{TIME_LIMIT}min.csv'
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Application", "Allocations", "Placement Time", "Solver Time", "Status"])
        for app, data in dict_allocations.items():
            writer.writerow([app] + list(data))

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()