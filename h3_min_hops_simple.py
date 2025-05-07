import sys
import os
import pandas as pd
import numpy as np
import random as rd
import copy
import time
import csv

# Configuração do caminho e ajustes globais
sys.path.append('modules')
np.set_printoptions(suppress=True)

# Variáveis globais
TOPOLOGY = 'synthetic'
DATA_PATH = 'melbourne2/'
COLUMNS = ['id', 'cpu', 'memory', 'storage', 'bandwidth']
CAPACITY_PERCENT = 1.0  # 100% da capacidade

# Importação de módulos personalizados
import get_data as gd  # type: ignore


# ----------------------------------------------------------------------------------

def get_data_nodes(topology, capacity_percent, columns):
    """
    Carrega os dados de nós, ajusta para a topologia 'melbourne' e remove a base station.
    """
    df_nodes, df_edges = gd.get_topology(topology, capacity_percent=capacity_percent)

    if topology == 'melbourne':
        df_nodes = df_nodes.iloc[1:].reset_index(drop=True)  # Remove o primeiro nó (Base Station)
        # Para topologia 'melbourne', todos os hops são iguais a 2
        num_nodes = len(df_nodes)
        hops_matrix = np.full((num_nodes, num_nodes), 2, dtype=int)
        np.fill_diagonal(hops_matrix, 0)  # Distância para si mesmo é 0
    else:
        graph = gd.create_graph_topology(df_nodes, df_edges)
        hops_matrix = gd.floyd_warshall(graph, weight='hops')

    return df_nodes[columns], df_edges, hops_matrix


def get_data_application(app_name, path, columns):
    """
    Carrega os dados de serviços e coreografias para uma aplicação.
    """
    df_services, df_choreog = gd.get_application(app_name, path)
    df_services = df_services[columns]
    df_services['id'] = range(len(df_services))  # Reindexar IDs dos serviços
    return df_services, df_choreog


def prepare_data_for_allocation(nodes, services):
    """
    Prepara os dados de nós e serviços para o formato de dicionário.
    """
    nodes_dict = {i: list(nodes.iloc[i, 1:]) for i in range(len(nodes))}
    services_dict = {i: list(services.iloc[i, 1:]) for i in range(len(services))}
    return nodes_dict, services_dict


# ----------------------------------------------------------------------------------

def count_hops_for_allocations(allocations, hops):
    """
    Conta o número total de hops entre os nós alocados.
    """
    return sum(hops[allocations[i]][allocations[i + 1]] for i in range(len(allocations) - 1))


def find_next_nodes(current_node, hops):
    """
    Encontra os nós vizinhos com base na matriz de hops.
    """
    hop_distances = hops[current_node]
    return sorted([(i, hop_distances[i]) for i in range(len(hop_distances)) if hop_distances[i] > 0], key=lambda x: x[1])


def has_sufficient_resources(service, node):
    """
    Verifica se um nó possui recursos suficientes para alocar o serviço.
    """
    return all(n >= s for n, s in zip(node, service))


def deduct_resources(node, service):
    """
    Deduz os recursos consumidos por um serviço de um nó.
    """
    return [n - s for n, s in zip(node, service)]


def allocate_services(services_dict, nodes_dict, hops_matrix):
    """
    Aloca serviços aos nós enquanto minimiza o número de hops.
    """
    best_allocations = []
    min_hops = float('inf')
    start_time = time.time()

    for starting_node in nodes_dict:
        allocations = []
        temp_nodes = copy.deepcopy(nodes_dict)  # Cópia temporária dos nós
        current_node = starting_node

        for service_id, service in services_dict.items():
            if has_sufficient_resources(service, temp_nodes[current_node]):
                allocations.append(current_node)
                temp_nodes[current_node] = deduct_resources(temp_nodes[current_node], service)
            else:
                next_nodes = find_next_nodes(current_node, hops_matrix)
                next_node = next((n[0] for n in next_nodes if has_sufficient_resources(service, temp_nodes[n[0]])), None)

                if next_node is None:
                    break  # Sem nó disponível para o serviço
                current_node = next_node
                allocations.append(current_node)
                temp_nodes[current_node] = deduct_resources(temp_nodes[current_node], service)

        if len(allocations) == len(services_dict):
            total_hops = count_hops_for_allocations(allocations, hops_matrix)
            if total_hops < min_hops:
                min_hops = total_hops
                best_allocations = [(s_id, n_id) for s_id, n_id in zip(services_dict.keys(), allocations)]

    allocation_time = '{:.8f}'.format(time.time() - start_time)
    
    return best_allocations, min_hops, [allocation_time]


# ----------------------------------------------------------------------------------

def main():
    # Carrega dados da topologia
    nodes, edges, hops_matrix = get_data_nodes(TOPOLOGY, CAPACITY_PERCENT, COLUMNS)
    allocations_dict = {}

    # Processa alocação para cada aplicação
    for app_index in range(20):
        app_name = f'App{app_index}'
        print(f'Processing {app_name}...')
        services, choreography = get_data_application(app_name, DATA_PATH, COLUMNS)

        nodes_dict, services_dict = prepare_data_for_allocation(nodes, services)

        best_allocations, min_hops, allocation_time = allocate_services(services_dict, nodes_dict, hops_matrix)

        print(f'Allocations: {best_allocations}')
        print(f'Minimum Hops: {min_hops}')
        # print(f'Allocation Time: {allocation_time:.8f} seconds\n')
        print(f'Allocation Time: {allocation_time[0]} s\n{"-" * 40}')


        allocations_dict[app_name] = (best_allocations, allocation_time)

    # Exporta resultados para CSV
    os.makedirs('results', exist_ok=True)
    output_file = f'results/h3_min_hops_{TOPOLOGY}_new.csv'
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, val in allocations_dict.items():
            writer.writerow([key, val])

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()