import sys
sys.path.append('modules')

import pandas as pd
import numpy as np 
import random as rd
import copy
import datetime
import get_data as gd  # type: ignore
import time
import csv

np.set_printoptions(suppress=True)

# ----------------------------------------------------------------------------------

# Global variables
global_topology = 'melbourne'
global_path = 'orign/'
global_columns = ['id', 'cpu', 'memory', 'storage', 'bandwidth']
global_capacity_percent = 1.0  # 100% capacity

# ----------------------------------------------------------------------------------

def get_data_nodes(topology, capacity_percent, columns):
    """
    Fetches topology nodes and edges, and returns the graph structure.
    """
    df_nodes, df_edges = gd.get_topology(topology, capacity_percent=capacity_percent)
    graph = gd.create_graph_topology(df_nodes, df_edges)
    return df_nodes[columns], df_edges, graph

def get_data_application(app_name, app_path, columns):
    """
    Fetches the service and choreography data for a given application.
    """
    df_services, df_choreog = gd.get_application(app_name, app_path)
    services_df = df_services[columns]
    services_df.loc[:, 'id'] = range(0, len(services_df))
    return services_df, df_choreog

def prepare_data_for_allocation(node_data, service_data):
    """
    Converts the node and service data into dictionary form for easier processing.
    """
    services_dict = {i: service_data[i][1:] for i in range(len(service_data))}
    nodes_dict = {i: node_data[i][1:] for i in range(len(node_data))}
    return nodes_dict, services_dict

# ----------------------------------------------------------------------------------

# Main Execution Loop
node_dataframe, edge_dataframe, graph_topology = get_data_nodes(global_topology, global_capacity_percent, global_columns)
hops_matrix = gd.floyd_warshall(graph_topology, weight='hops')

# Convert node data into an array of integers for further processing
node_array = node_dataframe.to_numpy().astype('int')

# ----------------------------------------------------------------------------------

def count_hops_for_allocations(allocations, hops):
    """
    Counts the total number of hops between the allocated nodes.
    """
    hops_counter = 0
    for n in range(len(allocations)-1):
        source_node = allocations[n]
        target_node = allocations[n+1]
        hops_counter += hops[source_node][target_node]
    return hops_counter

def find_next_nodes(current_node, hops):
    """
    Finds the neighboring nodes with hop counts.
    """
    hop_distances = hops[current_node]
    node_hop_pairs = [(i, hop_distances[i]) for i in range(len(hop_distances)) if hop_distances[i] != 0]
    return sorted(node_hop_pairs, key=lambda x: x[1])

def has_sufficient_resources(service_requirements, node_resources):
    """
    Checks if a node has sufficient resources to allocate the service.
    """
    return all(node_res >= service_req for node_res, service_req in zip(node_resources, service_requirements))

def find_next_available_node(service, candidate_nodes, nodes):
    """
    Finds the next node with sufficient resources to host the service.
    """
    for candidate in candidate_nodes:
        if has_sufficient_resources(service, nodes[candidate[0]]):
            return candidate
    return None

def deduct_service_resources_from_node(service_resources, node_resources):
    """
    Deducts the service resource requirements from the node's available resources.
    """
    return [node_res - service_res for node_res, service_res in zip(node_resources, service_resources)]

# ----------------------------------------------------------------------------------

def allocate_services(services_dict, nodes_dict, hops_matrix):
    """
    Allocates services to nodes while minimizing the number of hops.
    """
    final_allocations = []
    best_hop_count = float('inf')
    allocation_start_time = time.time()

    # Iterate through each node as a potential starting point
    for node_id, node_data in nodes_dict.items():
        allocated_nodes = []
        service_placements = []
        temp_nodes = copy.deepcopy(nodes_dict)  # Create a temporary copy of node resources

        current_node_id = node_id
        for service_id, service_data in services_dict.items():
            if has_sufficient_resources(service_data, temp_nodes[current_node_id]):
                allocated_nodes.append(current_node_id)
                service_placements.append([service_id, current_node_id])
                temp_nodes[current_node_id] = deduct_service_resources_from_node(service_data, temp_nodes[current_node_id])
            else:
                neighbor_nodes = find_next_nodes(current_node_id, hops_matrix)
                next_node = find_next_available_node(service_data, neighbor_nodes, temp_nodes)

                if not next_node:
                    break  # Stop if no suitable node is found
                current_node_id = next_node[0]
                allocated_nodes.append(current_node_id)
                service_placements.append([service_id, current_node_id])
                temp_nodes[current_node_id] = deduct_service_resources_from_node(service_data, temp_nodes[current_node_id])

        hop_count = count_hops_for_allocations(allocated_nodes, hops_matrix)
        if hop_count < best_hop_count and len(service_placements) == len(services_dict):
            best_hop_count = hop_count
            final_allocations = service_placements

    allocation_time_elapsed = time.time() - allocation_start_time
    return final_allocations, best_hop_count, allocation_time_elapsed

# ----------------------------------------------------------------------------------

# Dictionary to store allocations
allocations_dict = {}

# Loop through applications
for app_index in range(20):
    app_name = f'App{app_index}'
    print(f'Allocating services for {app_name}')
    
    services_data, choreography_data = get_data_application(app_name, global_path, global_columns)
    service_array = services_data.to_numpy().astype('int')

    nodes_for_allocation, services_for_allocation = prepare_data_for_allocation(node_array, service_array)

    # Call the allocation function
    best_allocations, min_hops, allocation_time = allocate_services(services_for_allocation, nodes_for_allocation, hops_matrix)

    print(f'Allocations: {best_allocations}')
    print(f'Time: {allocation_time:.2f} seconds')
    print(f'Minimum Hops: {min_hops}')
    print('\n-----------------------------------\n')

    allocations_dict[app_name] = (best_allocations, allocation_time)

# ----------------------------------------------------------------------------------

# Write allocations to CSV file
with open('results/h3_min_hops_'+global_topology+'.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    for app_name, allocation_data in allocations_dict.items():
        csv_writer.writerow([app_name, allocation_data])