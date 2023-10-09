import torch
from torch import Tensor
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import UPFD
from data_loader import FNNDataset

import networkx as nx

# function to construct the graph
def to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=False,
                remove_self_loops=False):
    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()
    G.add_nodes_from(range(data.num_nodes))
    node_attrs, edge_attrs = node_attrs or [], edge_attrs or []
    values = {}
    for key, item in data(*(node_attrs + edge_attrs)):
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]
    for i, (u, v) in enumerate(data.edge_index.t().tolist()):
        if to_undirected and v > u:
            continue
        if remove_self_loops and u == v:
            continue
        if u != v: 
            G.add_edge(u, v)
            for key in edge_attrs:
                G[u][v][key] = values[key][i]
    for key in node_attrs:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})
    return G
#function to print graphs
def print_graph(graph):
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(graph, seed=42)  # Posições dos nós usando layout spring
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=1000, font_size=10, font_color='black')
    plt.title(f'Graph Visualization - Sample {sample_id}')
    plt.show()


root_directory = ''  # dataset path
dataset_name = 'gossipcop'
feature_type = 'content'

#class call to build the intere dataset (train + test + val)
dataset = FNNDataset(root=root_directory, name=dataset_name, feature=feature_type)
sample_id = 0
print(dataset[sample_id].edge_index)
graph = to_networkx(dataset[sample_id])
print_graph(graph)
