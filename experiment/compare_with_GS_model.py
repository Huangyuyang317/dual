import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rnnisa.model import simulation, simu_opt
import numpy as np
from rnnisa.utils.tool_function import my_load
from time import time


def save_result_to_txt(pkl_name, runtime, optimal_cost, I_Sr, I_Se, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RNN Optimization Result\n")
        f.write("="*80 + "\n\n")
        f.write(f"File: {pkl_name}\n")
        f.write(f"Runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)\n")
        f.write(f"Optimal Cost: {optimal_cost:.3e}\n")
        f.write(f"\nI_Sr:\n{I_Sr.flatten()}\n")
        f.write(f"\nI_Se:\n{I_Se.flatten()}\n")
    print(f"Result saved to: {output_path}")


def run_rnn_optimization(data_type, nodes_num, sim, opt, I_S0_K, pkl_name):
    I_Sr0 = data_type(I_S0_K) * np.ones((1, nodes_num), dtype=data_type)
    I_Sr0[0, sim.get_demand_set()] = 40

    I_Se0 = data_type(0) * np.ones((1, nodes_num), dtype=data_type)
    raw_node_indices = np.nonzero(sim.get_raw_nodes()[0])[0]
    I_Se0[0, raw_node_indices] = (I_S0_K) * np.ones((1, len(raw_node_indices)), dtype=data_type) 
    
    t_start = time()
    _, _, I_Sr,I_Se = opt.two_stage_procedure(I_Sr0,I_Se0)
    runtime = time() - t_start
    
    if I_Sr.shape[0] < 20:
        print('I_Sr,I_Se (optimal inventory levels): ', I_Sr,I_Se)
    optimal_cost = sim.evaluate_cost(I_Sr=I_Sr,I_Se=I_Se, eval_num=100)  
    print('optimal cost of RNN-based method: %.3e' % optimal_cost)
    
    output_txt = f"./data/result_{pkl_name.replace('.pkl', '')}.txt"
    save_result_to_txt(pkl_name, runtime, optimal_cost, I_Sr, I_Se, output_txt)


def run_rnn_spanning_tree_optimization(data_type, temp_path, nodes_num, network_name):
    delivery_cycle_pkl_dict={10:'delivery_cycle-10nodes-2021-12-17 04-33.pkl'}
    step_size_dict={10:1.5e-1}
    step_size_e_dict={10:1.5e-2}
    regula_para_dict={10:1.5e3}
    stop_thresh_dict={10:1e-3}
    stop_thresh_ratio_dict={10:1.0}
    step_size_ratio_dict={10:0.3}
    step_bound_dict={10:None}
    I_S0_K_dict={10:20}
    raw_sr_regula_scale_dict={10:0.0}
    preserve_raw_stage2_dict={10:True}

    print(f"\n{'='*60}")
    print(f"Running: {network_name}")
    print(f"{'='*60}")

    sim = simulation.Simulation(data_type=data_type, duration=100, data_path=temp_path,
                                network_name=network_name,  
                                delivery_cycle=delivery_cycle_pkl_dict[nodes_num],
                                penalty_factor=2.0)
    opt = simu_opt.SimOpt(data_path=temp_path, rep_num=10, step_size=step_size_dict[nodes_num],step_size_e = step_size_e_dict[nodes_num],
                          regula_para=regula_para_dict[nodes_num],
                          stop_thresh=stop_thresh_dict[nodes_num], positive_flag=True, cost_f=sim.evaluate_cost,
                          grad_f=sim.evaluate_cost_gradient, step_bound=step_bound_dict[nodes_num],
                          stop_thresh_ratio=stop_thresh_ratio_dict[nodes_num],
                          step_size_ratio=step_size_ratio_dict[nodes_num], decay_mode=2, raw_nodes=sim.raw_node,
                          raw_sr_regula_scale=raw_sr_regula_scale_dict[nodes_num],
                          preserve_raw_stage2=preserve_raw_stage2_dict[nodes_num])
    run_rnn_optimization(data_type=data_type, nodes_num=nodes_num, sim=sim, opt=opt, I_S0_K=I_S0_K_dict[nodes_num], pkl_name=network_name)


if __name__ == "__main__":
    data_path = "./data"
    data_type = np.float64
    nodes_num_list = [10]
    
    pkl_files = [
        "bom_kodak_dual_lt_A_300.pkl"
    ]
    
    for pkl_file in pkl_files:
        for n in nodes_num_list:
            run_rnn_spanning_tree_optimization(data_type, data_path, n, pkl_file)
