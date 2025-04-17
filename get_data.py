import pandas as pd
import json

# Choosing the network topology used and importing the data
def get_topology(net_name, cluster=6, capacity_percent=1.0):
    df_nodes = pd.read_csv('data/topologies/'+net_name+'_nodes_capacity.csv')
    df_edges = pd.read_csv('data/topologies/'+net_name+'_edges_latency.csv')

    
    # cutting out the chosen cluster
    # just melbourne has clusters
    if net_name == 'melbourne':
        df_nodes = df_nodes[df_nodes['cluster'] == cluster]
        df_nodes = pd.DataFrame(df_nodes, columns=['id', 'x', 'y', 'cpu', 'memory', 'storage', 'bandwidth'])
        # df_edges = df_edges[df_edges['source'].isin(df_nodes['id']) | df_edges['target'].isin(df_nodes['id'])]
        df_edges = df_edges.loc[(df_edges['source'] == 6) | (df_edges['target'] == 6)]
        
        # restart index and column id of df_nodes
        df_nodes = df_nodes.reset_index(drop=True)

        for e, n in enumerate(df_nodes.values):
            df_edges.loc[df_edges['source'] == n[0], 'source'] = e
            df_edges.loc[df_edges['target'] == n[0], 'target'] = e

        df_nodes['id'] = df_nodes.index

    df_nodes['cpu'] = df_nodes['cpu'] * capacity_percent
    df_nodes['memory'] = df_nodes['memory'] * capacity_percent
    # df_nodes['storage'] = df_nodes['storage'] * capacity_percent
    df_nodes['bandwidth'] = df_nodes['bandwidth'] * capacity_percent
    df_edges['bandwidth'] = df_edges['bandwidth'] * capacity_percent

    return df_nodes, df_edges

# Choosing the application used and importing the data
def get_application(app_name, base):
    # Read JSON data from file

    with open('data/applications/'+base+'/'+app_name+'.json', 'r') as json_file:
        json_data = json_file.read()

    # Convert JSON data into a dictionary
    data_dict = json.loads(json_data)

    # Create DataFrames
    df_services = pd.DataFrame(data_dict['services'])

    # Expand choreography into a DataFrame
    choreography_list = []
    for app in data_dict['applications']:
        app_name = app['name']
        for payload in app['choreography']:
            payload_data = {
                'application': app_name,
                'source': payload['source'],
                'target': payload['target'],
                'payload': payload['payload'],
                'transmission_time': payload['transmission_time']
            }
            choreography_list.append(payload_data)

    df_choreog = pd.DataFrame(choreography_list)

    return df_services, df_choreog


import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

def create_graph_topology(df_net, df_edges):
    
    G = nx.Graph()

    # create a column named hop in df_edges 
    df_edges['hop'] = 1

    # Add nodes
    for i in range(len(df_net)):
        x = df_net.iloc[i, 1]
        y = df_net.iloc[i, 2]
        G.add_node(i,pos=(x,y))

    # Add edges
    for i in range(len(df_edges)):
        s = df_edges.iloc[i, 1]
        d = df_edges.iloc[i, 2]
        w = df_edges.iloc[i, 4]
        b = df_edges.iloc[i, 3]
        h = df_edges.iloc[i, 5]

        # format precision of weights
        w = float("{:.2f}".format(w))
        G.add_edge(s, d, latency=w, bandwidth=b, hop=h)

    return G

def floyd_warshall(G, weight='latency'):
    # get the shortest path 
    dists = nx.floyd_warshall_numpy(G, weight=weight)

    return dists

def floyd_warshall_all(G):
    # get the shortest path 
    dists_bw = nx.floyd_warshall_numpy(G, weight='bandwidth')
    dists_lt = nx.floyd_warshall_numpy(G, weight='latency')
    dists_hp = nx.floyd_warshall_numpy(G, weight='hop')

    return dists_bw, dists_lt, dists_hp