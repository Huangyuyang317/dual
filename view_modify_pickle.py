import pickle
import networkx as nx

file_path = "./data/bom_kodak_dual.pkl"

with open(file_path, 'rb') as f:
    graph = pickle.load(f)

adj_matrix = nx.adjacency_matrix(graph)
print("邻接矩阵形状:", adj_matrix.shape)
print("邻接矩阵:\n", adj_matrix.toarray())