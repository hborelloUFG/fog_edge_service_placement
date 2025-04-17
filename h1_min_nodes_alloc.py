import sys
sys.path.append('modules')

import pandas as pd
import numpy as np 
import random as rd

import datetime

# ----------------------------------------------------------------------------------
topology = 'melbourne'
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

def get_allocation_greed(nodes_data, services_data, T=1000):

    time_allocations = []
    best_allocations = []
    n_services = len(services_data)

    # parse values of services_data and nodes_data to int
    # nodes = [[int(i) for i in row] for row in nodes_data]
    # nodes = sorted(nodes, key=lambda x: (x[1], x[2], x[3], x[4]), reverse=True)

    start_time = time.time()

    # services receives services in random organization
    for i in range(T):
        services = [[int(i) for i in row] for row in services_data]
        nodes = [[int(i) for i in row] for row in nodes_data]
        nodes = sorted(nodes, key=lambda x: (x[1], x[2], x[3], x[4]), reverse=True)
        
        rd.shuffle(services)
        allocations = []
        allocated = []
        
        for n in nodes:
            for s in services:
                if n[1] >= s[1] and n[2] >= s[2] and n[3] >= s[3] and n[4] >= s[4]:
                    n[1] -= s[1]
                    n[2] -= s[2]
                    n[3] -= s[3]
                    n[4] -= s[4]

                    allocations.append([n[0], s[0]])
                    allocated.append(n[0])
                    services.remove(s)
            if services == []:
                # print('todos os serviços foram alocados!')
                break
        # print(i, len(set(allocated)), len(services))
        if n_services >= len(set(allocated)):
            best_allocations = allocations
            n_services = len(set(allocated))

    # print('allocations: ', allocations)
    time_allocations.append('{:.8f}'.format(time.time() - start_time))

    return best_allocations, time_allocations

nodes, edges, G = get_data_nodes()
dict_allocations = dict()

nodes_data = nodes.to_numpy().astype('int')

for i in range(20):
    application = 'App' + str(i)
    print('Allocation to ', application)
    services, coreog = get_data_application(application, path)
    services_data = services.to_numpy().astype('int')
    services_data = sorted(services_data, key=lambda s: max(s[1], s[2], s[3], s[4]), reverse=True)

    # def get_allocation_greed(nodes_data, services_data, T=1000):
    best_allocations, time_allocations = get_allocation_greed(nodes_data, services_data)

    # allocations, nodes_allocated, time_allocation, solver_time, status = get_allocation(services, nodes)
    print('Allocations : ', best_allocations)
    print('Time : ', time_allocations, 's')
    print('\n-----------------------------------\n')
    dict_allocations[application] = best_allocations, time_allocations


    # write dict_allocations to cvs file
# dict to csv
import csv
w = csv.writer(open('results/h1_min_nodes_alloc_'+topology+'.csv', 'w'))
for key, val in dict_allocations.items():
    w.writerow([key, val])