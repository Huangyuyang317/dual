import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import tensorflow as tf
from rnnisa.model import simulation

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def run_gradient_check():
    data_path = "./data"
    data_type = np.float32
    nodes_num = 10
    
    network_name_dict = {10: "bom_kodak_dual.pkl"}
    delivery_cycle_dict = {10: 'delivery_cycle-10nodes-2021-12-17 04-33.pkl'}
    
    sim = simulation.Simulation(
        data_type=data_type, 
        duration=100, 
        data_path=data_path,
        network_name=network_name_dict[nodes_num],
        delivery_cycle=delivery_cycle_dict[nodes_num],
        penalty_factor=20.0
    )

    I_Sr = 10.0 * np.ones((1, nodes_num), dtype=data_type)
    I_Se = 9.0 * np.ones((1, nodes_num), dtype=data_type)
    I_Se = I_Se * sim.get_raw_nodes()

    print(f"\n{'='*20} 梯度一致性校验 {'='*20}")
    
    for i in range(1):
        test_seed = 2026 + i
        print(f"\n[轮次 {i+1}] 测试随机种子: {test_seed}")

        # 获取打包好的标准参数
        args = sim.get_standard_args(I_Sr, I_Se, test_seed)

        from rnnisa.model.simulation import _simulate_and_bp_parallel
        cost_custom, grad_r_custom, grad_e_custom = _simulate_and_bp_parallel(args)

        from rnnisa.model.simulation import _simulate_and_bp_tf
        cost_tf, grad_r_tf, grad_e_tf = _simulate_and_bp_tf(args)

        cost_diff = abs(cost_custom - cost_tf)
        grad_r_custom = np.squeeze(grad_r_custom)
        grad_r_tf = np.squeeze(grad_r_tf)
        grad_e_custom = np.squeeze(grad_e_custom)
        if grad_e_tf is not None: grad_e_tf = np.squeeze(grad_e_tf)

        print(f"Cost 对比: 手动={cost_custom:.4f}, TF={cost_tf:.4f} | 差异={cost_diff:.2e}")
        print(f"Sr 手动梯度: \n{grad_r_custom}")
        print(f"Sr TF梯度: \n{grad_r_tf}")
        print(f"Sr 差异数组: \n{np.abs(grad_r_custom - grad_r_tf)}")
        print(f"Se 手动梯度: \n{grad_e_custom}")
        print(f"Se TF梯度: \n{grad_e_tf}")
        print(f"Se 差异数组: \n{np.abs(grad_e_custom - grad_e_tf)}")


        err_r = np.abs(grad_r_custom - grad_r_tf)
        max_err_r = np.max(err_r)
        
        if grad_e_custom is not None and grad_e_tf is not None:
            err_e = np.abs(grad_e_custom - grad_e_tf)
            max_err_e = np.max(err_e)
        else:
            max_err_e = 0.0
        print()
        print(f"Sr 梯度最大绝对误差: {max_err_r:.2e}")
        print(f"Se 梯度最大绝对误差: {max_err_e:.2e}")

        if max_err_r < 1e-4 and max_err_e < 1e-4:
            print("✅ 状态：梯度完全一致！你的算法逻辑正确。")
        else:
            print("❌ 状态：梯度不一致！请检查反向传播代码。")
            if nodes_num == 1:
                print(f"Sr 梯度详情 -> 手动: {grad_r_custom}, TF: {grad_r_tf}")
                print(f"Se 梯度详情 -> 手动: {grad_e_custom}, TF: {grad_e_tf}")

if __name__ == "__main__":
    run_gradient_check()