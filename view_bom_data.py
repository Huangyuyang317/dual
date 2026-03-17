"""
Script to load and display the contents of bom_kodak_dual.pkl
"""
import os
import sys
sys.path.append('c:/Users/HenryYoung/Desktop/dual')
from rnnisa.utils.tool_function import my_load
import networkx as nx
import numpy as np

def main():
    # Load the pickle file
    file_path = 'c:/Users/HenryYoung/Desktop/dual/data/bom_kodak_dual.pkl'
    print(f"Loading file: {file_path}")
    
    try:
        G = my_load(file_path)
        print(f"Successfully loaded object of type: {type(G)}")
        print()
        
        if isinstance(G, list):
            print(f'Object is a list with {len(G)} elements')
            G = G[0]  # Take the first element if it's a list
            print(f'Taking first element, new type: {type(G)}')
        else:
            print('Object is a single item')
        
        print()
        
        if isinstance(G, nx.Graph) or isinstance(G, nx.DiGraph):
            print('Graph Information:')
            print(f'- Number of nodes: {G.number_of_nodes()}')
            print(f'- Number of edges: {G.number_of_edges()}')
            print(f'- Is directed: {G.is_directed()}')
            
            print('\nNode attributes sample (first 5 nodes):')
            for i, (node, attrs) in enumerate(list(G.nodes(data=True))[:5]):
                print(f'  Node {node}: {attrs}')
            
            print('\nEdge attributes sample (first 5 edges):')
            for i, (u, v, attrs) in enumerate(list(G.edges(data=True))[:5]):
                print(f'  Edge ({u}, {v}): {attrs}')
                
            print('\nGraph properties:')
            print(f'- Nodes with holdcost attribute: {len(nx.get_node_attributes(G, "holdcost"))}')
            print(f'- Nodes with leadtime attribute: {len(nx.get_node_attributes(G, "leadtime"))}')
            print(f'- Nodes with mean attribute: {len(nx.get_node_attributes(G, "mean"))}')
            print(f'- Nodes with std attribute: {len(nx.get_node_attributes(G, "std"))}')
            
            # Show some statistics
            in_degrees = [v for k, v in G.in_degree()]
            out_degrees = [v for k, v in G.out_degree()]
            print('\nDegree statistics:')
            print(f'- In-degree values (first 10): {in_degrees[:10]}')
            print(f'- Out-degree values (first 10): {out_degrees[:10]}')
            print(f'- Nodes with in-degree 0 (demand nodes): {sum(1 for d in in_degrees if d == 0)}')
            print(f'- Nodes with out-degree 0 (raw material nodes): {sum(1 for d in out_degrees if d == 0)}')
            
            # Show adjacency matrix info
            adj_matrix = nx.adjacency_matrix(G, weight='weight')
            print(f'\nAdjacency matrix shape: {adj_matrix.shape}')
            print(f'Number of non-zero elements in adjacency matrix: {adj_matrix.nnz}')
            
        elif isinstance(G, dict):
            print('Dictionary keys:', list(G.keys()))
            for key, value in G.items():
                print(f'Key "{key}" type: {type(value)}, length: {len(value) if hasattr(value, "__len__") else "N/A"}')
        elif isinstance(G, (list, tuple)):
            print('List/Tuple length:', len(G))
            for i, item in enumerate(G):
                print(f'Item {i} type: {type(item)}, length: {len(item) if hasattr(item, "__len__") else "N/A"}')
        else:
            print('Object details:')
            print(f'- Type: {type(G)}')
            print(f'- Length (if applicable): {len(G) if hasattr(G, "__len__") else "N/A"}')
            print(f'- String representation: {str(G)[:500]}...')
            
    except Exception as e:
        print(f"Error loading file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()