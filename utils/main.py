import matplotlib.pyplot as plt
import networkx as nx
import readers as rw
import torch
from data_loader import FNNDataset
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import UPFD
from torch_geometric.nn import GCNConv


#function to generate graphs from dataset data
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
#function to print the graph
def print_graph(graph):
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(graph, seed=42)  # Posições dos nós usando layout spring
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=1000, font_size=10, font_color='black')
    plt.title(f'Graph Visualization - Sample {sample_id}')
    plt.show()



#getting the ids from dataset and the ids from fakenews from dataset to classify the graphs as t or f
path_news_id = ''
path_fake_ids=''
news_ids = rw.read_txt_to_array(path_news_id)
fake_ids = rw.read_csv_to_array(path_fake_ids)

#loading the dataset
root_directory_dataset = ''
dataset_name = 'gossipcop' #or polifact
feature_type = 'content'
data_exit_path = ''
dataset = FNNDataset(root=root_directory_dataset, name=dataset_name, feature=feature_type)
sample_id = 1
for sample_id in range(len(dataset)):
    graph = to_networkx(dataset[sample_id])
    #calculating node numbers
    numero_de_nos = len(graph.nodes)
    #calculating edge numbers
    numero_de_arestas = len(graph.edges)
    #calculating graph density
    densidade = 2 * len(graph.edges()) / (len(graph.nodes()) * (len(graph.nodes()) - 1))
    #calculating logest path
    longest_path = len(nx.algorithms.dag.dag_longest_path(graph))
    #calculating medium degree
    degree_values = dict(graph.degree())
    degree_values.pop(0)  # Remove o nó de posição 0 dos graus
    medium_degree = sum(degree_values.values()) / (len(graph) - 1)  
    #calculating biggest degree
    graus = dict(graph.degree())
    del graus[0] 
    biggest_degree = max(graus.values(), default=None)
    if(news_ids[sample_id] in fake_ids):
        flag = 'False'
    else:
        flag = 'True'
    degree_assortativity_in = nx.degree_assortativity_coefficient(graph, x='in')
    line = f'{flag};{numero_de_nos};{numero_de_arestas};{densidade};{longest_path};{medium_degree};{biggest_degree}'
    rw.write_to_csv('data_exit_path',line)