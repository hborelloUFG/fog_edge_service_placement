import sys
sys.path.append('modules')

import pandas as pd
import numpy as np 
import random
import copy
import time
import csv
import get_data as gd  # type: ignore

np.set_printoptions(suppress=True)

# ----------------------------------------------------------------------------------
topology = 'germany'
path = 'orign/'
columns = ['id', 'cpu', 'memory', 'storage', 'bandwidth']
cp = 1.0  # Capacity percent: 1.0 = 100%
alpha_min = 0.01  # Minimum 20% utilization
alpha_max = 0.99  # Maximum 80% utilization
tabu_tenure = 10  # Number of iterations before a move is removed from the Tabu list
max_iterations = 100  # Number of iterations for Tabu Search
# ----------------------------------------------------------------------------------

def get_data_nodes():
    df_nodes, df_edges = gd.get_topology(topology, capacity_percent=cp)
    G = gd.create_graph_topology(df_nodes, df_edges)
    return df_nodes[columns], df_edges, G

def get_data_application(app, path):
    df_services, df_choreog = gd.get_application(app, path)
    services = df_services[columns]
    services.loc[:, 'id'] = range(0, len(services))
    return services, df_choreog

def has_sufficient_resources(service_requirements, node_resources):
    """
    Checks if a node has sufficient resources to allocate the service.
    """
    return all(node_res >= service_req for node_res, service_req in zip(node_resources, service_requirements))

def deduct_service_resources_from_node(service_resources, node_resources):
    """
    Deducts the service resource requirements from the node's available resources.
    """
    return [node_res - service_res for node_res, service_res in zip(node_resources, service_resources)]

def check_capacity_constraints(allocations, nodes_dict, services_dict):
    """
    Validates if the allocation respects the alpha_min and alpha_max constraints for each node.
    """
    temp_nodes = copy.deepcopy(nodes_dict)
    for service_id, node_id in allocations:
        service_resources = services_dict.loc[service_id, ['cpu', 'memory', 'storage', 'bandwidth']]
        node_resources = temp_nodes.loc[node_id, ['cpu', 'memory', 'storage', 'bandwidth']]
        temp_nodes.loc[node_id, ['cpu', 'memory', 'storage', 'bandwidth']] = deduct_service_resources_from_node(service_resources, node_resources)

        # Check alpha limits
        total_resources = nodes_dict.loc[node_id, ['cpu', 'memory', 'storage', 'bandwidth']]
        used_resources = total_resources - temp_nodes.loc[node_id, ['cpu', 'memory', 'storage', 'bandwidth']]
        utilization = used_resources / total_resources
        if (utilization < alpha_min).any() or (utilization > alpha_max).any():
            return False
    return True

def count_hops_for_allocations(allocations, hops_matrix):
    """
    Counts the total number of hops between the allocated nodes for consecutive services.
    """
    hops_counter = 0
    for n in range(len(allocations) - 1):
        source_node = allocations[n][1]
        target_node = allocations[n + 1][1]
        hops_counter += hops_matrix[source_node, target_node]
    return hops_counter

def initial_random_allocation(services_dict, nodes_dict):
    """
    Generates a random initial allocation of services to nodes.
    """
    node_ids = list(nodes_dict.index)
    return [(service_id, random.choice(node_ids)) for service_id in services_dict.index]

def tabu_search(services, nodes, hops_matrix, max_iterations=100, tabu_tenure=10):
    """
    Tabu Search for minimizing hops and satisfying capacity constraints.
    """
    current_allocations = initial_random_allocation(services, nodes)
    best_allocations = current_allocations[:]
    tabu_list = []

    best_hop_count = count_hops_for_allocations(current_allocations, hops_matrix)
    current_hop_count = best_hop_count

    for iteration in range(max_iterations):
        neighborhood = []
        for i in range(len(current_allocations)):
            for j in range(i + 1, len(current_allocations)):
                new_allocations = current_allocations[:]
                new_allocations[i], new_allocations[j] = new_allocations[j], new_allocations[i]

                if check_capacity_constraints(new_allocations, nodes, services) and new_allocations not in tabu_list:
                    neighborhood.append(new_allocations)

        if not neighborhood:
            break

        best_neighbor = None
        best_neighbor_hop_count = float('inf')
        for neighbor in neighborhood:
            hop_count = count_hops_for_allocations(neighbor, hops_matrix)
            if hop_count < best_neighbor_hop_count:
                best_neighbor_hop_count = hop_count
                best_neighbor = neighbor

        if best_neighbor_hop_count < current_hop_count:
            current_allocations = best_neighbor
            current_hop_count = best_neighbor_hop_count
            tabu_list.append(best_neighbor)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
            if current_hop_count < best_hop_count:
                best_allocations = current_allocations[:]
                best_hop_count = current_hop_count

        print(f"Iteration {iteration+1}, Current Hops: {current_hop_count}, Best Hops: {best_hop_count}")

    return best_allocations, best_hop_count

# ----------------------------------------------------------------------------------

nodes, edges, G = get_data_nodes()
hops_matrix = gd.floyd_warshall(G, weight='hops')
dict_allocations = dict()

# Loop through applications
for i in range(20):
    application = f'App{i}'
    print(f'Allocation to {application}')
    services, choreog = get_data_application(application, path)

    # Run Tabu Search for the current application
    allocations, best_hop_count = tabu_search(services, nodes, hops_matrix, max_iterations=max_iterations, tabu_tenure=tabu_tenure)

    print(f'Best Allocation: {allocations}')
    print(f'Best Hop Count: {best_hop_count}\n')

    dict_allocations[application] = allocations, best_hop_count

# # Write allocations to CSV
# with open('tabu_search_allocations.csv', 'w', newline='') as csvfile:
#     csv_writer = csv.writer(csvfile)
#     for app_name, allocation_data in dict_allocations.items():
#         csv_writer.writerow([app_name, allocation_data])

print('Tabu Search Completed and Results Written to CSV.')