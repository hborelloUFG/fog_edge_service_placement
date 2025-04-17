import sys
sys.path.append('modules')

import pandas as pd
import numpy as np
import random
import copy
import get_data as gd  # type: ignore
import csv

np.set_printoptions(suppress=True)

# ----------------------------------------------------------------------------------
topology = 'germany'
path = 'orign/'
columns = ['id', 'cpu', 'memory', 'storage', 'bandwidth']
cp = 1.0  # Capacity percent: 1.0 = 100%
alpha_min = 0.2  # Minimum 20% utilization
alpha_max = 0.8  # Maximum 80% utilization
num_particles = 30  # Number of particles in the swarm
iterations = 100  # Number of iterations for PSO
c1 = 1.5  # Cognitive parameter
c2 = 1.5  # Social parameter
w = 0.5  # Inertia weight
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

# ----------------------------------------------------------------------------------
class Particle:
    def __init__(self, services, nodes):
        self.services = services
        self.nodes = nodes
        self.position = self.random_allocation()  # Current position (service-to-node allocation)
        self.velocity = [(0, 0)] * len(services)  # Initial velocity (unused in discrete space)
        self.best_position = self.position  # Personal best solution
        self.best_fitness = float('inf')  # Best fitness value

    def random_allocation(self):
        """
        Creates a random allocation of services to nodes.
        """
        node_ids = list(self.nodes.index)
        return [(service_id, random.choice(node_ids)) for service_id in self.services.index]

    def evaluate_fitness(self, hops_matrix, nodes_dict, services_dict):
        """
        Evaluates the fitness of the particle's current position.
        The fitness is the total number of hops if the allocation is valid, otherwise a large penalty.
        """
        if not check_capacity_constraints(self.position, nodes_dict, services_dict):
            return 1e6  # Large penalty for invalid solutions
        return count_hops_for_allocations(self.position, hops_matrix)

    def update_velocity(self, gbest_position):
        """
        Updates the particle's velocity based on its personal best and global best positions.
        PSO in discrete space typically doesn't use velocity, but you can update the position directly.
        """
        # No actual velocity update is needed for this problem, but we could explore swapping services between nodes
        pass

    def update_position(self, gbest_position):
        """
        Updates the particle's position by making small changes (swaps) based on its own best position and the global best.
        """
        for i in range(len(self.position)):
            if random.random() < c1:
                # Move towards personal best by swapping positions
                self.position[i] = self.best_position[i]
            if random.random() < c2:
                # Move towards global best by swapping positions
                self.position[i] = gbest_position[i]

        # After updating, we could repair the solution if it becomes invalid
        if not check_capacity_constraints(self.position, self.nodes, self.services):
            self.position = self.random_allocation()  # Revert to a random valid solution if invalid

# ----------------------------------------------------------------------------------

def pso(services, nodes, hops_matrix, num_particles=30, iterations=100):
    """
    Particle Swarm Optimization (PSO) to minimize hops and satisfy capacity constraints.
    """
    # Initialize the swarm
    particles = [Particle(services, nodes) for _ in range(num_particles)]
    gbest_position = None  # Global best position
    gbest_fitness = float('inf')  # Global best fitness

    for iteration in range(iterations):
        for particle in particles:
            # Evaluate fitness
            fitness = particle.evaluate_fitness(hops_matrix, nodes, services)

            # Update personal best
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position

            # Update global best
            if fitness < gbest_fitness:
                gbest_fitness = fitness
                gbest_position = particle.position

        # Update the velocity and position of each particle
        for particle in particles:
            particle.update_velocity(gbest_position)
            particle.update_position(gbest_position)

        print(f"Iteration {iteration+1}, Global Best Fitness (Min Hops): {gbest_fitness}")

    return gbest_position, gbest_fitness

# ----------------------------------------------------------------------------------

nodes, edges, G = get_data_nodes()
hops_matrix = gd.floyd_warshall(G, weight='hops')
dict_allocations = dict()

# Loop through applications
for i in range(4, 5):
    application = f'App{i}'
    print(f'Allocating services for {application}')
    services, choreog = get_data_application(application, path)

    # Run PSO for the current application
    best_allocation, best_fitness = pso(services, nodes, hops_matrix, num_particles, iterations)

    print(f'Best Allocation after PSO: {best_allocation}')
    print(f'Best Fitness (Min Hops): {best_fitness}\n')

    dict_allocations[application] = best_allocation, best_fitness

# Write allocations to CSV