import numpy as np
from scipy.stats import norm

def solve_dual_source_dp():
    # ================= 配置参数 =================
    duration = 100
    le, lr = 1, 2
    ce, cs = 2.0, 0.0   # 加急费 2.0, 普通费 0.0
    h = 5.0             # 持有成本
    p = 100.0             # 缺货罚金 (penalty_factor=2)
    gamma = 1        # 折扣因子 (接近1表示关注长期平均成本)
    
    # 需求分布 (正态分布离散化)
    mu, sigma = 2, 0.5
    max_demand = int(mu + 4 * sigma)
    demand_range = np.arange(0, max_demand + 1)
    demand_probs = norm.pdf(demand_range, mu, sigma)
    demand_probs /= demand_probs.sum()

    # 状态空间定义
    # x: 净库存 (在手 - 缺货)
    # q1: 预计在下一个周期(t+1)到达的订单量
    min_x, max_x = -20, 40
    min_q, max_q = 0, 30
    
    x_grid = np.arange(min_x, max_x + 1)
    q_grid = np.arange(min_q, max_q + 1)
    
    nx, nq = len(x_grid), len(q_grid)
    V = np.zeros((nx, nq)) # 值函数矩阵
    
    # 决策空间 (下单量必须是非负整数)
    max_order = 20
    order_range = np.arange(0, max_order + 1)

    print("正在进行值迭代...")

    # ================= 值迭代循环 =================
    for it in range(50): # 迭代50次通常足以观察收敛趋势
        new_V = np.copy(V)
        for i, x in enumerate(x_grid):
            for j, q1 in enumerate(q_grid):
                
                best_val = float('inf')
                
                # 遍历可能的决策 (Oe, Or)
                for Oe in order_range:
                    for Or in order_range:
                        # 1. 计算当期期望成本 (持有 + 缺货)
                        # 注意：本期的成本由 x + q1 (即本期期初可用量) 决定
                        expected_holding_penalty = 0
                        for d, prob in zip(demand_range, demand_probs):
                            inventory_after_demand = x + q1 - d
                            cost = h * max(0, inventory_after_demand) + p * max(0, -inventory_after_demand)
                            expected_holding_penalty += prob * cost
                        
                        current_cost = expected_holding_penalty + ce * Oe + cs * Or
                        
                        # 2. 计算未来期望价值
                        # 下一期的状态: x' = x + q1 - d, q1' = Oe + Or (因为Lr=2, Or下期变成q1; Le=1, Oe下期到货)
                        future_val = 0
                        for d, prob in zip(demand_range, demand_probs):
                            next_x = np.clip(x + q1 - d, min_x, max_x)
                            next_q1 = np.clip(Oe + Or, min_q, max_q)
                            
                            idx_x = next_x - min_x
                            idx_q = next_q1 - min_q
                            future_val += prob * V[idx_x, idx_q]
                        
                        total_val = current_cost + gamma * future_val
                        
                        if total_val < best_val:
                            best_val = total_val
                
                new_V[i, j] = best_val
        
        diff = np.abs(new_V - V).max()
        V = new_V
        if diff < 1e-2:
            break
# ================= 策略提取 =================
    print("迭代完成。正在分析最优策略结构...")
    
    # 建立两个矩阵，存储每个状态下的最佳下单决策
    policy_Oe = np.zeros((nx, nq))
    policy_Or = np.zeros((nx, nq))

    for i, x in enumerate(x_grid):
        for j, q1 in enumerate(q_grid):
            best_val = float('inf')
            best_decision = (0, 0)
            
            # 重新寻找让 V(x, q1) 最小的 (Oe, Or)
            for Oe in order_range:
                for Or in order_range:
                    # (计算过程同值迭代内部...)
                    inventory_after_demand = x + q1 - mu # 用均值近似观察
                    current_cost = h * max(0, inventory_after_demand) + p * max(0, -inventory_after_demand) + ce * Oe + cs * Or
                    
                    next_x = np.clip(x + q1 - mu, min_x, max_x)
                    next_q1 = np.clip(Oe + Or, min_q, max_q)
                    total_val = current_cost + gamma * V[next_x - min_x, next_q1 - min_q]
                    
                    if total_val < best_val:
                        best_val = total_val
                        best_decision = (Oe, Or)
            
            policy_Oe[i, j] = best_decision[0]
            policy_Or[i, j] = best_decision[1]

    # --- 反推 Se ---
    # 根据 DIP 定义：当 IP_e < Se 时，下加急单 Oe
    # 在这个 DP 模型里，加急提前期是 1，所以 IP_e = x + q1
    # 我们找 Oe > 0 的临界点
    possible_Se = []
    for i, x in enumerate(x_grid):
        for j, q1 in enumerate(q_grid):
            if policy_Oe[i, j] > 0:
                possible_Se.append(x + q1 + policy_Oe[i, j]) 
    
    Se_star = np.max(possible_Se) if possible_Se else 0

    # --- 反推 Sr ---
    # 根据 DIP 定义：当 IP_total < Sr 时，下普通单 Or
    # IP_total = x + q1 + Oe (加急单下完后立刻计入总位置)
    possible_Sr = []
    for i, x in enumerate(x_grid):
        for j, q1 in enumerate(q_grid):
            if policy_Or[i, j] > 0:
                possible_Sr.append(x + q1 + policy_Oe[i, j] + policy_Or[i, j])

    Sr_star = np.max(possible_Sr) if possible_Sr else 0

    print(f"理论反推的最优加急水位 Se: {Se_star}")
    print(f"理论反推的最优常规水位 Sr: {Sr_star}")
if __name__ == "__main__":
    solve_dual_source_dp()