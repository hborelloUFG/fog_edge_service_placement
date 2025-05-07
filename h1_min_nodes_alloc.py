# modules/main.py

import sys
sys.path.append('modules')

import pandas as pd
import numpy as np
import random as rd
import time
import csv

import get_data as gd  # type: ignore

topology = 'germany'
path = 'orign/'
columns = ['id', 'cpu', 'memory', 'storage', 'bandwidth']
cp = 1.0

np.set_printoptions(suppress=True)

# Funções
def get_data_nodes():
    try:
        df_nodes, df_edges = gd.get_topology(topology, capacity_percent=cp)
        G = gd.create_graph_topology(df_nodes, df_edges)
        return df_nodes[columns], df_edges, G
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar dados dos nós: {e}")

def get_data_application(app, path):
    try:
        df_services, df_choreog = gd.get_application(app, path)
        services = df_services[columns]
        services.loc[:, 'id'] = range(0, len(services))
        return services, df_choreog
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar dados da aplicação {app}: {e}")

# Heurística: executa exatamente 100 vezes, armazena número de nós alocados, calcula estatísticas e tempos
def get_allocation_greed(nodes_data, services_data):
    NUM_RUNS = 100
    nodes_allocated_list = []
    time_list = []
    best_num_nodes = None
    best_allocations = []
    n_services = len(services_data)

    for run in range(NUM_RUNS):
        start_time = time.time()
        services = [[int(i) for i in row] for row in services_data]
        nodes = [[int(i) for i in row] for row in nodes_data]
        
        nodes = sorted(nodes, key=lambda x: (x[1], x[2], x[3], x[4]), reverse=True)
        rd.shuffle(services)

        allocations = []
        allocated = []

        # Passo 1: tentar alocar todos os serviços em um único nó.
        for n in nodes:
            if all(n[i] >= sum(s[i] for s in services) for i in range(1, 5)):
                for s in services:
                    n[1:5] = [n[i] - s[i] for i in range(1, 5)]
                    allocations.append([s[0], n[0]])
                allocated.append(n[0])
                break

        # Passo 2: se não possível, alocar serviço a serviço.
        if len(allocations) != len(services_data):
            allocations = []
            for n in nodes:
                for s in services[:]:
                    if all(n[i] >= s[i] for i in range(1, 5)):
                        n[1:5] = [n[i] - s[i] for i in range(1, 5)]
                        allocations.append([s[0], n[0]])
                        allocated.append(n[0])
                        services.remove(s)
                if not services:
                    break

        num_nodes = len(set(allocated))
        nodes_allocated_list.append(num_nodes)
        time_list.append(time.time() - start_time)
        # Armazena a melhor alocação (menor número de nós)
        if (best_num_nodes is None) or (num_nodes < best_num_nodes):
            best_num_nodes = num_nodes
            best_allocations = allocations

    min_nodes = min(nodes_allocated_list)
    mean_nodes = float(np.mean(nodes_allocated_list))
    std_nodes = float(np.std(nodes_allocated_list))
    mean_time = float(np.mean(time_list))
    return min_nodes, mean_nodes, std_nodes, mean_time, best_allocations

# Execução
def main():
    nodes, edges, G = get_data_nodes()
    dict_allocations = dict()
    nodes_data = nodes.to_numpy().astype('int')

    results_summary = []

    for i in range(20):
        application = f"App{i}"
        services, coreog = get_data_application(application, path)
        print(f"Allocation to {application} with {len(services)} services")

        services_data = services.to_numpy().astype('int')
        services_data = sorted(services_data, key=lambda s: max(s[1:5]), reverse=True)

        min_nodes, mean_nodes, std_nodes, mean_time, best_allocations = get_allocation_greed(nodes_data, services_data)
        print(f"Menor número de nós alocados: {min_nodes}")
        print(f"Média de nós alocados: {mean_nodes:.2f}")
        print(f"Desvio padrão da alocação de nós: {std_nodes:.2f}")
        print(f"Média dos tempos de decisão: {mean_time:.8f} s")
        print('-' * 40)
        dict_allocations[application] = (min_nodes, mean_nodes, std_nodes, mean_time, best_allocations)
        results_summary.append([application, min_nodes, mean_nodes, std_nodes, mean_time, best_allocations])

    # Salva resultados detalhados (opcional)
    with open(f'results/mid/h1_min_nodes_alloc_{topology}_new.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Application', 'MinNodes', 'MeanNodes', 'StdNodes', 'MeanTime', 'BestAllocations'])
        for row in results_summary:
            writer.writerow(row)

    # Imprime resumo final para cada aplicação
    print("\nResumo dos resultados por aplicação:")
    for row in results_summary:
        print(f"Aplicação: {row[0]} | Menor nº de nós: {row[1]} | Média de nós: {row[2]:.2f} | Desvio padrão: {row[3]:.2f} | Média tempo: {row[4]:.8f} s")

if __name__ == "__main__":
    main()