"""
This module contains the experiments for the RNN inspired simulation approach for large-scale inventory optimization problems
discussed in the paper, "Large-Scale Inventory Optimization: A Recurrent-Neural-Networks-Inspired Simulation Approach".

Author:
    Tan Wang
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rnnisa.model import simulation, simu_opt
import numpy as np
from rnnisa.utils.tool_function import my_load


def run_rnn_optimization(data_type, nodes_num, sim, opt, I_S0_K):
    I_Sr0 = data_type(I_S0_K) * np.ones((1, nodes_num), dtype=data_type)
    I_Sr0[0, sim.get_demand_set()] = 4

    I_Se0 = data_type(0) * np.ones((1, nodes_num), dtype=data_type)
    raw_node_indices = np.nonzero(sim.get_raw_nodes()[0])[0]  # 获取值为True/非零的索引
    I_Se0[0, raw_node_indices] = (I_S0_K) * np.ones((1, len(raw_node_indices)), dtype=data_type) 
    
    _, _, I_Sr,I_Se = opt.two_stage_procedure(I_Sr0,I_Se0)
    if I_Sr.shape[0] < 20:
        print('I_Sr,I_Se (optimal inventory levels): ', I_Sr,I_Se)
    optimal_cost = sim.evaluate_cost(I_Sr=I_Sr,I_Se=I_Se, eval_num=100)  
    print('optimal cost of RNN-based method: %.3e' % optimal_cost)


def run_rnn_spanning_tree_optimization(data_type, temp_path, nodes_num):
    network_name_dict={1:"bom_single_node.pkl"}
    delivery_cycle_pkl_dict={1:'delivery_cycle-1node-2026-03-22.pkl'}
    step_size_dict={1:1e-2}
    regula_para_dict={1:0}
    stop_thresh_dict={1:2.1e-4}
    stop_thresh_ratio_dict={1:0.74}
    step_size_ratio_dict={1:0.09}
    step_bound_dict={1:None}
    I_S0_K_dict={1:3}

    sim = simulation.Simulation(data_type=data_type, duration=100, data_path=temp_path,
                                network_name=network_name_dict[nodes_num],  
                                delivery_cycle=delivery_cycle_pkl_dict[nodes_num],
                                penalty_factor=20)
    opt = simu_opt.SimOpt(data_path=temp_path, rep_num=10, step_size=step_size_dict[nodes_num],
                          regula_para=regula_para_dict[nodes_num],
                          stop_thresh=stop_thresh_dict[nodes_num], positive_flag=True, cost_f=sim.evaluate_cost,
                          grad_f=sim.evaluate_cost_gradient, step_bound=step_bound_dict[nodes_num],
                          stop_thresh_ratio=stop_thresh_ratio_dict[nodes_num],
                          step_size_ratio=step_size_ratio_dict[nodes_num], decay_mode=2, raw_nodes=sim.raw_node)
    run_rnn_optimization(data_type=data_type, nodes_num=nodes_num, sim=sim, opt=opt, I_S0_K=I_S0_K_dict[nodes_num])


if __name__ == "__main__":
    data_path = "./data"
    data_type = np.float32
    nodes_num_list = [1]
    for n in nodes_num_list:
        run_rnn_spanning_tree_optimization(data_type, data_path, n)