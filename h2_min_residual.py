import sys
sys.path.append('modules')

import pandas as pd
import numpy as np 
import random as rd
import copy
import os

import datetime

# ----------------------------------------------------------------------------------
topology = 'germany'
path = 'orign/'
columns = ['id', 'cpu', 'memory', 'storage', 'bandwidth']

# capacity percent: 0.8 = 80%
cp = 1.0
# ----------------------------------------------------------------------------------
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory

import time
np.set_printoptions(suppress=True)

import get_data as gd # type: ignore

def get_data_nodes():
    df_nodes, df_edges = gd.get_topology(topology, capacity_percent=cp)
    G = gd.create_graph_topology(df_nodes, df_edges)

    return df_nodes[columns], df_edges, G

def get_data_application(app, path):
    df_services, df_choreog = gd.get_application(app, path)

    services = df_services[columns]
    services.loc[:, 'id'] = range(0, len(services))

    return services, df_choreog


def get_nodes_utilization(lines):
    nodes_utilization = []

    for app_key, allocations in lines.items():
        # Obtém os serviços da aplicação
        services_placed, _ = get_data_application(app_key, path)

        # Obtém os índices dos nós alocados
        allocated_node_indices = list(set([a[1] for a in allocations]))

        # Obtém os dados dos nós alocados
        nodes_allocated_data = [nodes[n] for n in allocated_node_indices]

        # Soma dos recursos dos serviços e dos nós (ignorando o ID)
        total_service_resources = np.sum(services_placed.to_numpy().astype('int'), axis=0)
        total_node_resources = np.sum(nodes_allocated_data, axis=0)

        utilization = np.sum(total_service_resources[1:]) / np.sum(total_node_resources[1:])
        nodes_utilization.append(utilization)

    keys = list(lines.keys())
    return nodes_utilization, keys

def get_allocation(nodes_data, services_data, app):

    import copy
    MAX_TRIALS = 100
    T = min(MAX_TRIALS, 10 * len(services_data))
    no_improvement = 0
    MAX_NO_IMPROVEMENT = 20

    all_best_allocations = []
    all_best_utilizations = 0.0
    all_mean_time = None
    all_mean_utilization = 0.0
    all_std_utilization = 0.0

    for k in range(MAX_TRIALS):

        # Inicializa variáveis para armazenar a melhor alocação
        best_allocations = []
        best_utilization = 0.0
        time_allocations = []
        all_utilizations = []
        mean_time = None

        start_time = time.time()
        for t in range(T):
            # Recria os dados para manter independência entre execuções
            # temp_services = copy.deepcopy(services_data)
            # temp_nodes = copy.deepcopy(nodes_data)

            temp_services = services_data.copy()
            temp_nodes = nodes_data.copy()

            # Inicializa listas para cada execução
            allocated_nodes = []
            allocated_services = []
            allocations = []

            # Embaralha os serviços
            rd.shuffle(temp_services)

            for i in range(len(temp_services)):
                if i in allocated_services:
                    continue

                cpu, mem, sto, bw = temp_services[i][1:]
                able_nodes = []

                for j in range(len(temp_nodes)):
                    if j in allocated_nodes:
                        continue

                    node_cpu, node_mem, node_sto, node_bw = temp_nodes[j][1:]
                    if cpu <= node_cpu and mem <= node_mem and sto <= node_sto and bw <= node_bw:
                        able_nodes.append(j)

                if len(able_nodes) != 0:
                    allocated_node = able_nodes[0]
                    allocated_nodes.append(allocated_node)
                    allocated_services.append(i)
                    allocations.append([i, allocated_node])

                    temp_nodes[allocated_node][1] -= cpu
                    temp_nodes[allocated_node][2] -= mem
                    temp_nodes[allocated_node][3] -= sto
                    temp_nodes[allocated_node][4] -= bw

                    for k in range(i+1, len(temp_services)):
                        if k in allocated_services:
                            continue

                        cpu_k, mem_k, sto_k, bw_k = temp_services[k][1:]
                        node_cpu, node_mem, node_sto, node_bw = temp_nodes[allocated_node][1:]

                        if cpu_k <= node_cpu and mem_k <= node_mem and sto_k <= node_sto and bw_k <= node_bw:
                            allocated_services.append(k)
                            allocations.append([k, allocated_node])

                            temp_nodes[allocated_node][1] -= cpu_k
                            temp_nodes[allocated_node][2] -= mem_k
                            temp_nodes[allocated_node][3] -= sto_k
                            temp_nodes[allocated_node][4] -= bw_k

            time_allocations.append('{:.8f}'.format(time.time() - start_time))
            dict_allocations = {app: allocations}
            utilization, _ = get_nodes_utilization(dict_allocations)
            all_utilizations.append(utilization[0])

            if utilization[0] > best_utilization:
                best_utilization = utilization[0]
                best_allocations = allocations
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= MAX_NO_IMPROVEMENT:
                    break
        
        mean_time = np.mean([float(t) for t in time_allocations])
        # calcule a media de utilização e o seu desvio padrão
        mean_utilization = np.mean([float(u) for u in all_utilizations])
        std_utilization = np.std([float(u) for u in all_utilizations])

    if best_utilization > all_best_utilizations:
        all_best_allocations = best_allocations
        all_best_utilizations = best_utilization
        all_mean_time = mean_time
        all_mean_utilization = mean_utilization
        all_std_utilization = std_utilization
    
    return all_best_allocations, all_mean_time, all_best_utilizations, all_mean_utilization, all_std_utilization

nodes, edges, G = get_data_nodes()
dict_allocations = dict()
allocations_dict = dict()

nodes = nodes.to_numpy().astype('int')

for i in range(20):
    application = 'App' + str(i)
    # print('Allocation to ', application, path)
    services, coreog = get_data_application(application, path)
    print('Allocation to ', application, 'with', len(services), 'services')

    services_data = services.to_numpy().astype('int')
    services_data = sorted(services_data, key=lambda s: max(s[1], s[2], s[3], s[4]), reverse=True)

    nodes_data = copy.deepcopy(nodes)

    # def get_allocation_greed(nodes_data, services_data, T=1000):
    best_allocations, time_allocations, best_utilization, mean_utilization, std_utilization = get_allocation(nodes_data, services_data, application)

    # allocations, nodes_allocated, time_allocation, solver_time, status = get_allocation(services, nodes)
    print('BEST Utilization : ', best_utilization)
    print('Mean Utilization : ', mean_utilization)
    print('Std Utilization : ', std_utilization)
    print('Time : ', time_allocations, 's')
    print('\n-----------------------------------\n')
    dict_allocations[application] = best_allocations, time_allocations
    # write dict_allocations to cvs file

    allocations_dict[application] = [best_utilization, mean_utilization, std_utilization, time_allocations, best_allocations]


# Exporta resultados para CSV
import csv
os.makedirs('results', exist_ok=True)
output_file = f'results/mid/h2_min_residual_{topology}_new.csv'
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Application', 'MinNodes', 'MeanNodes', 'StdNodes', 'MeanTime', 'BestAllocations'])
    for key, val in allocations_dict.items():
        min_nodes, mean_nodes, std_nodes, mean_time, best_allocations = val
        writer.writerow([key, min_nodes, mean_nodes, std_nodes, mean_time, str(best_allocations)])

print(f"Results saved to {output_file}")

# # dict to csv
# w = csv.writer(open('results/mid/h2_min_residual_'+topology+'_new.csv', 'w'))
# for key, val in dict_allocations.items():
#     w.writerow([key, val])
