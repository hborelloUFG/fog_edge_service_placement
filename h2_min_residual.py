import sys
import pandas as pd
import numpy as np
import random
import copy
import time
import csv
from typing import Tuple, List, Dict

# Adicionar caminho do módulo
sys.path.append('modules')

# Configurações globais
TOPOLOGY = 'synthetic'
PATH = 'melbourne2/'
COLUMNS = ['id', 'cpu', 'memory', 'storage', 'bandwidth']
CAPACITY_PERCENT = 1.0  # 100%

# Importar módulo personalizado
import get_data as gd  # type: ignore

# --------------------------------------------
# Funções Auxiliares
# --------------------------------------------

def get_data_nodes() -> Tuple[pd.DataFrame, pd.DataFrame, any]:
    """
    Obtém os nós e arestas da topologia.
    """
    df_nodes, df_edges = gd.get_topology(TOPOLOGY, capacity_percent=CAPACITY_PERCENT)
    graph = gd.create_graph_topology(df_nodes, df_edges)
    return df_nodes[COLUMNS], df_edges, graph


def get_data_application(app: str, path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Obtém os serviços e a coreografia para uma aplicação específica.
    """
    df_services, df_choreography = gd.get_application(app, path)
    df_services = df_services[COLUMNS]
    df_services['id'] = range(len(df_services))  # Redefinir IDs sequenciais
    return df_services, df_choreography

def get_allocation(nodes_data: np.ndarray, services_data: np.ndarray) -> Tuple[List[List[int]], List[str]]:
    """
    Aloca serviços nos nós com base nos recursos residuais, executando T tentativas com embaralhamento.
    Retorna a melhor alocação encontrada.
    """
    T = 10 * len(services_data)  # Número de tentativas
    best_allocations = []
    best_remaining_resources = float('inf')
    best_elapsed_time = None

    start_time = time.time()

    for _ in range(T):
        nodes_copy = copy.deepcopy(nodes_data)
        np.random.shuffle(nodes_copy)  # Embaralhar nós antes da alocação
        allocated_nodes = set()
        allocations = []

        for service_id, cpu, mem, sto, bw in services_data:
            # Filtrar nós capazes de alocar o serviço
            capable_nodes = [
                node_id for node_id, node_cpu, node_mem, node_sto, node_bw in nodes_copy
                if node_id not in allocated_nodes and
                cpu <= node_cpu and mem <= node_mem and sto <= node_sto and bw <= node_bw
            ]

            if capable_nodes:
                allocated_node = capable_nodes[0]  # Selecionar o primeiro nó capaz
                allocated_nodes.add(allocated_node)
                allocations.append([service_id, allocated_node])

                # Atualizar recursos do nó
                node_idx = np.where(nodes_copy[:, 0] == allocated_node)[0][0]
                nodes_copy[node_idx, 1:] -= [cpu, mem, sto, bw]

        # Calcular recursos residuais
        total_remaining_resources = nodes_copy[:, 1:].sum()
        
        # Atualizar melhor solução se necessário
        if total_remaining_resources < best_remaining_resources:
            best_remaining_resources = total_remaining_resources
            best_allocations = allocations
            best_elapsed_time = '{:.8f}'.format(time.time() - start_time)

    return best_allocations, [best_elapsed_time]


def write_to_csv(filename: str, data):
    """
    Escreve as alocações em um arquivo CSV no formato esperado.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for key, val in data.items():
            writer.writerow([key, val])


# --------------------------------------------
# Execução Principal
# --------------------------------------------

def main():
    # Obter dados de nós e arestas
    nodes_df, edges_df, graph = get_data_nodes()
    nodes = nodes_df.to_numpy().astype('int')

    # Inicializar dicionário de alocações
    dict_allocations = {}

    for i in range(20):  # Processar 20 aplicações
        app_name = f'App{i}'
        services_df, choreography_df = get_data_application(app_name, PATH)
        print(f'Processing {app_name} with {len(services_df)} services...')

        # Preparar dados dos serviços
        services_data = services_df.to_numpy().astype('int')
        services_data = sorted(services_data, key=lambda s: max(s[1:]), reverse=True)

        # Alocar serviços nos nós
        nodes_copy = copy.deepcopy(nodes)
        allocations, elapsed_time = get_allocation(nodes_copy, services_data)

        # Exibir resultados
        print(f'Allocations: {allocations}')
        print(f'Time: {elapsed_time[0]} s\n{"-" * 40}')

        # Armazenar alocações
        dict_allocations[app_name] = (allocations, elapsed_time)

    # Salvar resultados em CSV no formato original
    output_file = f'results/h2_min_residual_{TOPOLOGY}_new.csv'
    write_to_csv(output_file, dict_allocations)
    print(f'Results saved to {output_file}')


if __name__ == "__main__":
    main()