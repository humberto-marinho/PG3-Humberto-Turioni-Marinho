import torch
from torch import Tensor
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import UPFD
from data_loader import FNNDataset
import readers as rw
import networkx as nx

""" # From PyG utils #function witout edges in the same vertice
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
        if u != v:  # Verifica se u e v são diferentes antes de adicionar a aresta
            G.add_edge(u, v)
            for key in edge_attrs:
                G[u][v][key] = values[key][i]
    for key in node_attrs:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})
    return G """
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
        G.add_edge(u, v)
        for key in edge_attrs:
            G[u][v][key] = values[key][i]
    for key in node_attrs:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})
    return G
def print_graph(graph):
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(graph, seed=42)  # Posições dos nós usando layout spring
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=1000, font_size=10, font_color='black')
    plt.title(f'Graph Visualization - Sample {sample_id}')
    plt.show()

#train_data = UPFD(root="C:\\Users\\user\\Desktop\\execucao", name="gossipcop", feature="content", split="test")

#getting the ids from dataset and the ids from fakenews from dataset to classify the graphs as t or f
news_ids_gossicop = rw.read_txt_to_array('C:\\Users\\user\\Desktop\\tcc2\\GNN-FakeNews\\data\\gos_news_list.txt')
fake_ids_gossicop = rw.read_csv_to_array('C:\\Users\\user\\Desktop\\tcc2\\GNN-FakeNews\\data\\ids_gossipcop_fake.csv')
print(news_ids_gossicop[0])


#loading the dataset
root_directory = 'C:\\Users\\user\\Desktop\\tcc2\\GNN-FakeNews\\data\\UPFD\\'  # Substitua pelo caminho correto
dataset_name = 'gossipcop'
feature_type = 'content'
dataset = FNNDataset(root=root_directory, name=dataset_name, feature=feature_type)
sample_id = 0
graph = to_networkx(dataset[sample_id])
import networkx as nx

# Calculando a densidade do grafo
density = nx.density(graph)
# Calculando o número de componentes fortemente conectados
num_strongly_connected_components = nx.number_strongly_connected_components(graph)
# Calculando a assortatividade direcionada por grau de entrada
degree_assortativity_in = nx.degree_assortativity_coefficient(graph, x='in')
# Calculando a centralidade de intermediação global
betweenness_centrality_global = nx.betweenness_centrality(graph)
# Calculando o coeficiente de clustering direcionado
clustering_coefficient = nx.average_clustering(graph.to_undirected())
# Calculando o número total de ciclos no grafo
num_cycles = len(list(nx.simple_cycles(graph)))
# Calculando a transitividade direcionada
transitivity = nx.transitivity(graph)
# Calculando a assimetria do grafo
asymmetry = nx.overall_reciprocity(graph)
# Calculando a assortatividade direcionada
directed_assortativity = nx.assortativity.degree_assortativity_coefficient(graph, x='in', y='out')
line = 0
print(line)
""" G_excluindo_origem = graph.copy()
# Remova o nó de origem
G_excluindo_origem.remove_node(0)
# Calcule o maior grau no grafo excluindo a origem
maior_grau = max(dict(G_excluindo_origem.degree()).values())
print(maior_grau) """
# show graph using Matplotlib
