import sys
sys.path.append('modules')

import pandas as pd
import numpy as np 
import random as rd
import copy

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

def get_allocation(nodes_data, services_data):

    # create a list to store the nodes that are already allocated
    allocated_nodes = []
    allocated_services = []
    allocations =  []
    time_allocations = []

    start_time = time.time()

    # iterate over the services
    for i in range(len(services_data)):
        # check if the service has already been allocated
        if i in allocated_services:
            continue

        # get the requirements of the current service
        cpu, mem, sto, bw = services_data[i][1:]

        # create a list to store the nodes that can accommodate the current service
        able_nodes = []

        # iterate over the nodes
        for j in range(len(nodes_data)):
            # check if the node is already allocated
            if j in allocated_nodes:
                continue

            # get the capacity of the current node
            node_cpu, node_mem, node_sto, node_bw = nodes_data[j][1:]

            # check if the node can accommodate the current service
            if cpu <= node_cpu and mem <= node_mem and sto <= node_sto and bw <= node_bw:
                able_nodes.append(j)

        # check if there are any nodes that can accommodate the current service
        if len(able_nodes) != 0:
            # allocate the service to the first able node
            allocated_node = able_nodes[0]
            allocated_nodes.append(allocated_node)
            allocated_services.append(i)
            # print(f"Service {i} allocated to node {allocated_node}")
            allocations.append([allocated_node, i])


            # subtract the node capabilities
            nodes_data[allocated_node][1] -= cpu
            nodes_data[allocated_node][2] -= mem
            nodes_data[allocated_node][3] -= sto
            nodes_data[allocated_node][4] -= bw

            # check if there are any subsequent services that can be allocated to the same node
            for k in range(i+1, len(services_data)):
                # check if the service has already been allocated
                if k in allocated_services:
                    continue

                # get the requirements of the current service
                cpu, mem, sto, bw = services_data[k][1:]

                # get the capacity of the allocated node
                node_cpu, node_mem, node_sto, node_bw = nodes_data[allocated_node][1:]

                # check if the node can accommodate the current service
                if cpu <= node_cpu and mem <= node_mem and sto <= node_sto and bw <= node_bw:
                    allocated_services.append(k)
                    # print(f"Service {k} allocated to node {allocated_node}")
                    allocations.append([allocated_node, k])

                    # subtract the node capabilities
                    nodes_data[allocated_node][1] -= cpu
                    nodes_data[allocated_node][2] -= mem
                    nodes_data[allocated_node][3] -= sto
                    nodes_data[allocated_node][4] -= bw

    time_allocations.append('{:.8f}'.format(time.time() - start_time))

    return allocations, time_allocations

nodes, edges, G = get_data_nodes()
dict_allocations = dict()

nodes = nodes.to_numpy().astype('int')

for i in range(20):
    application = 'App' + str(i)
    print('Allocation to ', application)
    services, coreog = get_data_application(application, path)
    services_data = services.to_numpy().astype('int')
    services_data = sorted(services_data, key=lambda s: max(s[1], s[2], s[3], s[4]), reverse=True)

    nodes_data = copy.deepcopy(nodes)

    # def get_allocation_greed(nodes_data, services_data, T=1000):
    best_allocations, time_allocations = get_allocation(nodes_data, services_data)

    # allocations, nodes_allocated, time_allocation, solver_time, status = get_allocation(services, nodes)
    print('Allocations : ', best_allocations)
    print('Time : ', time_allocations, 's')
    print('\n-----------------------------------\n')
    dict_allocations[application] = best_allocations, time_allocations


    # write dict_allocations to cvs file
# dict to csv
import csv
w = csv.writer(open('results/h2_min_residual_'+topology+'.csv', 'w'))
for key, val in dict_allocations.items():
    w.writerow([key, val])