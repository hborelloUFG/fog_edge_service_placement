# modules/main.py

import sys
sys.path.append('modules')

import pandas as pd
import numpy as np
import random as rd
import datetime
import time
import csv

import get_data as gd  # type: ignore

# Configurações
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

def get_allocation_greed(nodes_data, services_data, T=10, convergence_limit=5):
    best_allocations = []
    n_services = len(services_data)
    time_allocations = []

    start_time = time.time()
    no_improvement_count = 0  # Contador de iterações sem melhoria


    for _ in range(len(services_data)*T):
        services = [[int(i) for i in row] for row in services_data]
        nodes = [[int(i) for i in row] for row in nodes_data]
        nodes = sorted(nodes, key=lambda x: (x[1], x[2], x[3], x[4]), reverse=True)  # Nós com mais recursos primeiro
        rd.shuffle(services)

        allocations = []
        allocated = []

        # **Passo 1**: Tentar alocar todos os serviços em um único nó.
        for n in nodes:
            if all(n[i] >= sum(s[i] for s in services) for i in range(1, 5)):
                # Atualiza os recursos do nó.
                for s in services:
                    n[1:5] = [n[i] - s[i] for i in range(1, 5)]
                    allocations.append([s[0], n[0]])
                allocated.append(n[0])
                break

        # **Passo 2**: Caso não seja possível alocar em um único nó, fazer alocação serviço a serviço.
        if len(allocations) != len(services_data):
            allocations = []  # Reinicia as alocações
            for n in nodes:
                for s in services[:]:  # Itera sobre uma cópia para remover alocados
                    if all(n[i] >= s[i] for i in range(1, 5)):  # Verifica recursos
                        n[1:5] = [n[i] - s[i] for i in range(1, 5)]  # Atualiza os recursos do nó
                        allocations.append([s[0], n[0]])
                        allocated.append(n[0])
                        services.remove(s)  # Remove o serviço alocado
                if not services:
                    break

        # **Passo 3**: Atualizar a melhor alocação se necessário.
        if n_services >= len(set(allocated)):
            best_allocations = allocations
            n_services = len(set(allocated))
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # Interrompe se a solução não melhorar após várias iterações
        if no_improvement_count >= (convergence_limit*n_services):
            no_improvement_count = 0
            break

    time_allocations.append(f"{time.time() - start_time:.8f}")
    print(f"T: {len(best_allocations)}")
    return best_allocations, time_allocations

# Execução
def main():
    nodes, edges, G = get_data_nodes()
    dict_allocations = dict()

    nodes_data = nodes.to_numpy().astype('int')

    for i in range(20):
        application = f"App{i}"
        services, coreog = get_data_application(application, path)
        print(f"Allocation to {application} with {len(services)} services")

        services_data = services.to_numpy().astype('int')
        services_data = sorted(services_data, key=lambda s: max(s[1:5]), reverse=True)

        best_allocations, time_allocations = get_allocation_greed(nodes_data, services_data)
        print(f"Allocations: {best_allocations}")
        print(f"Time: {time_allocations} s")
        n_nodes = len(set([x[1] for x in best_allocations]))
        print(f"Number of allocated nodes: {n_nodes}\n{'-' * 35}\n")
        dict_allocations[application] = (best_allocations, time_allocations)

    with open(f'results/h1_min_nodes_alloc_{topology}_new.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for key, val in dict_allocations.items():
            writer.writerow([key, val])

if __name__ == "__main__":
    main()