import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import tensorflow as tf
from rnnisa.model import simulation

# 禁用 GPU 以免 TF 精度受环境影响，并保持与 NumPy 对齐
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def run_gradient_check():
    # --- 1. 环境准备 ---
    data_path = "./data"
    data_type = np.float32
    nodes_num = 1 # 先用单节点测试，成功后再改 10
    
    # 模拟 single.py 的参数设置
    network_name_dict = {1: "bom_single_node.pkl"}
    delivery_cycle_dict = {1: 'delivery_cycle-1node-2026-03-22.pkl'}
    
    sim = simulation.Simulation(
        data_type=data_type, 
        duration=100, 
        data_path=data_path,
        network_name=network_name_dict[nodes_num],
        delivery_cycle=delivery_cycle_dict[nodes_num],
        penalty_factor=20.0
    )

    # --- 2. 设定测试点 (Sr, Se) ---
    I_Sr = 10.0 * np.ones((1, nodes_num), dtype=data_type)
    I_Se = 5.0 * np.ones((1, nodes_num), dtype=data_type)
    I_Se = I_Se * sim.get_raw_nodes() # 只有原材料节点 Se 非 0

    print(f"\n{'='*20} 梯度一致性校验 {'='*20}")
    
    # 连续测试 3 次不同的随机种子
    for i in range(10):
        test_seed = 2026 + i
        print(f"\n[轮次 {i+1}] 测试随机种子: {test_seed}")

        # 获取打包好的标准参数
        args = sim.get_standard_args(I_Sr, I_Se, test_seed)

        # --- A. 调用你的算法 (手动 BP/IPA) ---
        # 直接调用底层的并行函数（单进程运行方便打印）
        from rnnisa.model.simulation import _simulate_and_bp_parallel
        cost_custom, grad_r_custom, grad_e_custom = _simulate_and_bp_parallel(args[0])

        # --- B. 调用 TensorFlow 自动微分 ---
        from rnnisa.model.simulation import _simulate_and_bp_tf
        cost_tf, grad_r_tf, grad_e_tf = _simulate_and_bp_tf(args[1])

        # --- 3. 结果对比分析 ---
        cost_diff = abs(cost_custom - cost_tf)
        # 去掉多余维度进行对比
        grad_r_custom = np.squeeze(grad_r_custom)
        grad_r_tf = np.squeeze(grad_r_tf)
        grad_e_custom = np.squeeze(grad_e_custom)
        if grad_e_tf is not None: grad_e_tf = np.squeeze(grad_e_tf)

        print(f"Cost 对比: 手动={cost_custom:.4f}, TF={cost_tf:.4f} | 差异={cost_diff:.2e}")
        print(f"Sr 梯度对比: 手动={grad_r_custom:.4f}, TF={grad_r_tf:.4f} | 差异={np.abs(grad_r_custom - grad_r_tf):.2e}")
        print(f"Se 梯度对比: 手动={grad_e_custom:.4f}, TF={grad_e_tf:.4f} | 差异={np.abs(grad_e_custom - grad_e_tf):.2e}")
        
        # 计算 Sr 梯度误差
        err_r = np.abs(grad_r_custom - grad_r_tf)
        max_err_r = np.max(err_r)
        
        # 计算 Se 梯度误差 (如果有)
        if grad_e_custom is not None and grad_e_tf is not None:
            err_e = np.abs(grad_e_custom - grad_e_tf)
            max_err_e = np.max(err_e)
        else:
            max_err_e = 0.0
        print()
        print(f"Sr 梯度最大绝对误差: {max_err_r:.2e}")
        print(f"Se 梯度最大绝对误差: {max_err_e:.2e}")

        # 判定标准：通常 1e-4 以内视为算法正确
        if max_err_r < 1e-4 and max_err_e < 1e-4:
            print("✅ 状态：梯度完全一致！你的算法逻辑正确。")
        else:
            print("❌ 状态：梯度不一致！请检查反向传播代码。")
            # 打印具体数值辅助 Debug
            if nodes_num == 1:
                print(f"Sr 梯度详情 -> 手动: {grad_r_custom}, TF: {grad_r_tf}")
                print(f"Se 梯度详情 -> 手动: {grad_e_custom}, TF: {grad_e_tf}")

if __name__ == "__main__":
    run_gradient_check()