import sys
sys.path.append('modules')

import pandas as pd
import numpy as np 
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

def get_placement(services, nodes, s_matrix, time_limit=30):

    qtd_nodes = len(nodes)
    qtd_services = len(services)
    resources = ['cpu', 'memory', 'storage', 'bandwidth']  # k = 1 to 4 (resource types)
    
    start_time = time.time()

    model = pyo.ConcreteModel()

    # Define variables
    model.x = pyo.Var(range(qtd_services), range(qtd_nodes), domain=Binary)  # Service placement
    model.u = pyo.Var(range(1, qtd_services), range(qtd_nodes), range(qtd_nodes), domain=Binary)  # Hops between consecutive services

    x = model.x
    u = model.u

    # Objective function: Minimize the number of hops between nodes for consecutive services
    def hops_minimization_rule(model):
        return sum(s_matrix[j1, j2] * u[i, j1, j2] 
                   for i in range(1, qtd_services) 
                   for j1 in range(qtd_nodes) 
                   for j2 in range(qtd_nodes))
    
    model.obj = pyo.Objective(rule=hops_minimization_rule, sense=pyo.minimize)

    # Constraint 1: Resource usage per node should not exceed its capacity
    model.con1 = pyo.ConstraintList()
    for j in range(qtd_nodes):
        for res in resources:
            model.con1.add(sum(x[i, j] * services.loc[i, res] for i in range(qtd_services)) <= nodes.loc[j, res])

    # Constraint 2: Each service must be placed on exactly one node
    model.con2 = pyo.ConstraintList()
    for i in range(qtd_services):
        model.con2.add(sum(x[i, j] for j in range(qtd_nodes)) == 1)

    # Constraint 3: Consecutive services can only have a hop between nodes if they are placed on different nodes
    model.con3 = pyo.ConstraintList()
    for i in range(1, qtd_services):  # From service 2 to m
        for j1 in range(qtd_nodes):
            for j2 in range(qtd_nodes):
                model.con3.add(x[i-1, j1] + x[i, j2] <= u[i, j1, j2] + 1)  # Linking hop with placement

    # Solve the model using Gurobi or any other available solver
    opt = SolverFactory('gurobi', solver_io="python")

    # Set time to get the solution in 30 minutes
    opt.options['TimeLimit'] = 60 * time_limit
    results = opt.solve(model)

    time_placement = '{:.8f}'.format(time.time() - start_time)
    time_solver = results.solver.wallclock_time

    # Retrieve the result: List of node and service combinations
    nodes_allocated = []
    allocations = []
    for i in range(qtd_services):
        for j in range(qtd_nodes):
            if model.x[i, j]() == 1:
                nodes_allocated.append(j)
                allocations.append((i, j))

    # Display the results
    print('Objective (min hops) : ', model.obj())
    print('Allocations : ', allocations)
    print('Time : ', time_placement, 's')

    # get execution time of solver
    print('Time Solver = ', time_solver, 's')

    # Print solver status
    termination_condition = results.solver.termination_condition
    status = 'Infeasible'
    if termination_condition == TerminationCondition.optimal:
        status = 'Optimal'
    elif termination_condition == TerminationCondition.feasible:
        status = 'Feasible'
    print('Status : ', status)

    return allocations, nodes_allocated, time_placement, time_solver, status

nodes, edges, G = get_data_nodes()
s_matrix = gd.floyd_warshall(G, weight='hops')
dict_allocations = dict()

for i in range(20):
    application = 'App' + str(i)
    print('Allocation to ', application)
    services, coreog = get_data_application(application, path)
    allocations, nodes_allocated, time_placement, solver_time, status = get_placement(services, nodes, s_matrix)

    print('\n-----------------------------------\n')
    dict_allocations[application] = allocations, time_placement, solver_time, status

# dict to csv
import csv
w = csv.writer(open('m3_min_hops_pyomo.csv', 'w'))
for key, val in dict_allocations.items():
    w.writerow([key, val])
