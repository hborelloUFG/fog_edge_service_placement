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
TOPOLOGY = 'germany'
PATH = 'orign/'
COLUMNS = ['id', 'cpu', 'memory', 'storage', 'bandwidth']
CAPACITY_PERCENT = 1.0  # 100%
NODE_PENALTY_WEIGHT = 5  # Peso ajustável para penalizar uso de múltiplos nós

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
        # np.random.shuffle(nodes_copy)  # Embaralhar nós antes da alocação
        sort_idx = np.lexsort((-nodes_copy[:, 4], -nodes_copy[:, 3], -nodes_copy[:, 2], -nodes_copy[:, 1]))
        nodes_copy = nodes_copy[sort_idx]

        # Embaralhar serviços antes da alocação
        np.random.shuffle(services_data)

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
                # Selecionar o nó que resulta no menor custo composto (recursos residuais + penalização por número de nós alocados)
                min_residual = float('inf')
                best_node = None

                for node_id in capable_nodes:
                    node_idx = np.where(nodes_copy[:, 0] == node_id)[0][0]
                    cpu_residual = nodes_copy[node_idx, 1] - cpu
                    mem_residual = nodes_copy[node_idx, 2] - mem
                    sto_residual = nodes_copy[node_idx, 3] - sto
                    bw_residual  = nodes_copy[node_idx, 4] - bw
                    # Penalização pelo número de nós já alocados
                    residual_sum = cpu_residual + mem_residual + sto_residual + bw_residual + NODE_PENALTY_WEIGHT * len(allocated_nodes)

                    if residual_sum < min_residual:
                        min_residual = residual_sum
                        best_node = node_id

                allocated_nodes.add(best_node)
                allocations.append([service_id, best_node])

                # Atualizar recursos do nó selecionado
                node_idx = np.where(nodes_copy[:, 0] == best_node)[0][0]
                nodes_copy[node_idx, 1:] -= [cpu, mem, sto, bw]

        # Calcular recursos residuais
        total_remaining_resources = nodes_copy[:, 1:].sum()
        
        # Atualizar melhor solução se necessário
        if total_remaining_resources < best_remaining_resources:
            best_remaining_resources = total_remaining_resources
            best_allocations = allocations
            best_elapsed_time = '{:.8f}'.format(time.time() - start_time)

    return best_remaining_resources, best_allocations, [best_elapsed_time]


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
    nodes_allocated_list = []

    for i in range(20):  # Processar 20 aplicações
        app_name = f'App{i}'
        services_df, choreography_df = get_data_application(app_name, PATH)
        print(f'Processing {app_name} with {len(services_df)} services...')

        # Preparar dados dos serviços
        services_data = services_df.to_numpy().astype('int')
        services_data = sorted(services_data, key=lambda s: max(s[1:]), reverse=True)

        # Alocar serviços nos nós
        nodes_copy = copy.deepcopy(nodes)
        remaining_resources, allocations, elapsed_time = get_allocation(nodes_copy, services_data)

        # Exibir resultados
        print(f'Allocations: {allocations}')
        print(f'Remaining resources: {remaining_resources}')
        print(f'Time: {elapsed_time[0]} s\n{"-" * 40}')

        # Criar uma lista
        nodes_allocated_list.append([remaining_resources])

        # Armazenar alocações
        dict_allocations[app_name] = (allocations, elapsed_time)

    # Salvar resultados em CSV no formato original
    output_file = f'results/h2_min_residual_{TOPOLOGY}_new.csv'
    write_to_csv(output_file, dict_allocations)
    print(f'Results saved to {output_file}')

    print(f'Residual: {nodes_allocated_list}')


if __name__ == "__main__":
    main()


    # Residual: [[53191], [54894], [52964], [53922], [63717], [53739], [61770], [57221], [54712], [54254], [54204], [63992], [58268], [54843], [50379], [59143], [57738], [53792], [58187], [60004]]
    # Residual: [[53191], [54894], [52964], [53922], [63717], [53739], [61770], [57221], [54712], [54254], [54204], [63992], [58268], [54843], [50379], [59143], [57738], [53792], [58187], [60004]]
    # Residual: [[53191], [54894], [52964], [53922], [63717], [53739], [61770], [57221], [54712], [54254], [54204], [63992], [58268], [54843], [50379], [59143], [57738], [53792], [58187], [60004]]
    # Residual: [[53191], [54894], [52964], [53922], [63717], [53739], [61770], [57221], [54712], [54254], [54204], [63992], [58268], [54843], [50379], [59143], [57738], [53792], [58187], [60004]]