import pickle
import networkx as nx
import numpy as np

file_path = "./data/bom_kodak_dual.pkl"

with open(file_path, 'rb') as f:
    graph = pickle.load(f)

print("=" * 60)
print("Network Structure of bom_kodak_dual.pkl")
print("=" * 60)

print(f"\nGraph type: {type(graph)}")
print(f"Number of nodes: {graph.number_of_nodes()}")
print(f"Number of edges: {graph.number_of_edges()}")

print("\n" + "=" * 60)
print("Nodes and their attributes:")
print("=" * 60)

for node in sorted(graph.nodes()):
    attrs = dict(graph.nodes[node])
    print(f"\nNode '{node}':")
    for attr, value in attrs.items():
        print(f"  {attr}: {value}")

print("\n" + "=" * 60)
print("Edges and their weights:")
print("=" * 60)

for edge in sorted(graph.edges()):
    weight = graph[edge[0]][edge[1]].get('weight', 'N/A')
    print(f"  {edge[0]} -> {edge[1]}: weight = {weight}")

print("\n" + "=" * 60)
print("Summary of key attributes:")
print("=" * 60)

is_dual = [graph.nodes[n].get('is_dual_source', False) for n in graph.nodes()]
lt_fast = [graph.nodes[n].get('lt_fast', 'N/A') for n in graph.nodes()]
lt_slow = [graph.nodes[n].get('lt_slow', 'N/A') for n in graph.nodes()]
cost_fast = [graph.nodes[n].get('cost_fast', 'N/A') for n in graph.nodes()]
cost_slow = [graph.nodes[n].get('cost_slow', 'N/A') for n in graph.nodes()]

print(f"\nNodes with is_dual_source=True:")
for n in graph.nodes():
    if graph.nodes[n].get('is_dual_source', False):
        print(f"  {n}: lt_fast={graph.nodes[n].get('lt_fast')}, lt_slow={graph.nodes[n].get('lt_slow')}, cost_fast={graph.nodes[n].get('cost_fast')}, cost_slow={graph.nodes[n].get('cost_slow')}")

print(f"\nAll nodes lt_fast: {lt_fast}")
print(f"All nodes lt_slow: {lt_slow}")
print(f"All nodes cost_fast: {cost_fast}")
print(f"All nodes cost_slow: {cost_slow}")