import pickle

source_file = "./data/bom_kodak_dual.pkl"
output_file = "./data/bom_kodak_dual_lt_A_300.pkl"

with open(source_file, 'rb') as f:
    graph = pickle.load(f)

holdcost_values = {
    'A': 1,
    'B': 3,
    'C': 4,
    'D': 6,
    'E': 12,
    'F': 20,
    'G': 13,
    'H': 8,
    'I': 1,
    'J': 50,
}

lt_fast_values = {
    'A': 1,
}

lt_slow_values = {
    'A': 6,
}

cost_fast_values = {
    
}

cost_slow_values = {
    # 'A': 5,   # 如果需要修改在这里添加
}
# =======================================

for node, hc in holdcost_values.items():
    if node in graph.nodes:
        graph.nodes[node]['holdcost'] = hc

for node, val in lt_fast_values.items():
    if node in graph.nodes:
        graph.nodes[node]['lt_fast'] = val

for node, val in lt_slow_values.items():
    if node in graph.nodes:
        graph.nodes[node]['lt_slow'] = val

for node, val in cost_fast_values.items():
    if node in graph.nodes:
        graph.nodes[node]['cost_fast'] = val

for node, val in cost_slow_values.items():
    if node in graph.nodes:
        graph.nodes[node]['cost_slow'] = val

with open(output_file, 'wb') as f:
    pickle.dump(graph, f)

print(f"已保存: {output_file}")
print("\n当前参数:")
print("-"*60)
print(f"{'节点':<8} {'lt_fast':<10} {'lt_slow':<10} {'holdcost':<12} {'cost_fast':<12} {'cost_slow':<12}")
print("-"*60)
for node in sorted(graph.nodes()):
    attrs = graph.nodes[node]
    print(f"{node:<8} {attrs.get('lt_fast'):<10} {attrs.get('lt_slow'):<10} {attrs.get('holdcost'):<12} {attrs.get('cost_fast'):<12} {attrs.get('cost_slow'):<12}")