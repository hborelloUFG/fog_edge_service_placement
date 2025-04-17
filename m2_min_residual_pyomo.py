import sys
sys.path.append('modules')

import pandas as pd
import numpy as np 
import datetime

# ----------------------------------------------------------------------------------
topology = 'synthetic'
path = 'melbourne2/'
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

def get_placement(services, nodes):

    qtd_nodes = len(nodes)
    qtd_services = len(services)
    resources = ['cpu', 'memory', 'storage', 'bandwidth']  # k = 1 to 4 (resource types)
    
    start_time = time.time()

    model = pyo.ConcreteModel()

    # define variables x(num_services, num_nodes) and z(num_nodes) with binary values
    model.x = pyo.Var(range(qtd_services), range(qtd_nodes), domain=Binary)  # Service placement
    model.z = pyo.Var(range(qtd_nodes), domain=Binary)                      # Node activation

    x = model.x
    z = model.z

    # Objective function: Minimize residual capacity on allocated nodes
    def residual_capacity_rule(model):
        return sum(
            sum((nodes.loc[j, res] * z[j] - sum(x[i, j] * services.loc[i, res] for i in range(qtd_services))) 
                for res in resources)
            for j in range(qtd_nodes)
        )
    model.obj = pyo.Objective(rule=residual_capacity_rule, sense=pyo.minimize)

    # Constraint 1: Resource usage per node should not exceed its capacity
    model.con1 = pyo.ConstraintList()
    for j in range(qtd_nodes):
        for res in resources:
            model.con1.add(sum(x[i, j] * services.loc[i, res] for i in range(qtd_services)) <= nodes.loc[j, res])

    # Constraint 2: Each service must be placed on exactly one node
    model.con2 = pyo.ConstraintList()
    for i in range(qtd_services):
        model.con2.add(sum(x[i, j] for j in range(qtd_nodes)) == 1)

    # Constraint 3: A node is activated if at least one service is placed on it
    model.con3 = pyo.ConstraintList()
    for j in range(qtd_nodes):
        model.con3.add(sum(x[i, j] for i in range(qtd_services)) <= qtd_services * z[j])

    # Solve the model using Gurobi or any other available solver
    opt = SolverFactory('gurobi', solver_io="python")
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
dict_allocations = dict()

for i in range(20):
    application = 'App' + str(i)
    services, coreog = get_data_application(application, path)
    print('Allocation to ', application, 'with', len(services), 'services.')
    allocations, nodes_allocated, time_placement, solver_time, status = get_placement(services, nodes)

    print('\n-----------------------------------\n')
    dict_allocations[application] = allocations, time_placement, solver_time, status

# write dict_allocations to cvs file
# dict to csv
import csv
w = csv.writer(open('results/m2_min_residual_pyomo_'+topology+'_new.csv', 'w'))
for key, val in dict_allocations.items():
    w.writerow([key, val])
