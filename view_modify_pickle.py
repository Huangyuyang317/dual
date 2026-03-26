import pickle
import networkx as nx

def load_and_view_pickle(file_path):
    """加载并查看pickle文件的内容"""
    with open(file_path, 'rb') as f:
        graph = pickle.load(f)
    
    print("Graph type:", type(graph))
    print("Nodes:", list(graph.nodes()))
    print("Edges:", list(graph.edges()))
    
    for node in graph.nodes():
        print(f"\nNode {node}:")
        for attr, value in graph.nodes[node].items():
            print(f"  {attr}: {value}")
    
    return graph

def modify_graph(graph, modifications):
    """
    修改图的节点属性
    modifications: dict, 格式为 {node: {attribute: new_value}}
    """
    for node, attrs in modifications.items():
        for attr, value in attrs.items():
            graph.nodes[node][attr] = value
    return graph

def save_graph(graph, file_path):
    """保存图到pickle文件"""
    with open(file_path, 'wb') as f:
        pickle.dump(graph, f)

if __name__ == "__main__":
    file_path = "./data/bom_single_node.pkl"
    
    # 加载并查看原始文件
    print("Original graph:")
    graph = load_and_view_pickle(file_path)
    
    # 显示修改选项
    print("\nCurrent attributes for node A:")
    for attr, value in graph.nodes['A'].items():
        print(f"  {attr}: {value}")
    
    print("\nWhat would you like to modify?")
    print("Available attributes: leadtime, holdcost, lt_fast, lt_slow, cost_fast, cost_slow, is_dual_source, mean, std")
    
    # 示例修改 - 你可以在这里指定你想要的修改
    # 这里是一个示例，假设我们想修改一些值
    modifications = {
        'A': {
            'cost_fast': 20,  
        }
    }
    
    # 应用修改
    modified_graph = modify_graph(graph, modifications)
    
    print("\nAfter modifications:")
    for attr, value in modified_graph.nodes['A'].items():
        print(f"  {attr}: {value}")
    
    # 询问是否要保存修改
    save_choice = input("\nDo you want to save these modifications to the file? (y/n): ")
    if save_choice.lower() == 'y':
        backup_path = file_path.replace('.pkl', '_backup.pkl')
        # 先备份原文件
        original_graph = load_and_view_pickle(file_path)
        with open(backup_path, 'wb') as f:
            pickle.dump(original_graph, f)
        print(f"Backup saved to {backup_path}")
        
        # 保存修改后的文件
        save_graph(modified_graph, file_path)
        print(f"Modifications saved to {file_path}")
        
        # 验证修改
        print("\nVerifying the saved file:")
        load_and_view_pickle(file_path)