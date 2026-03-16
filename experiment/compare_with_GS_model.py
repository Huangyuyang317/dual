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
    I_S0 = data_type(I_S0_K) * np.ones((1, nodes_num), dtype=data_type)
    I_S0[0, sim.get_demand_set()] = 40
    
    _, I_S = opt.two_stage_procedure(I_S0)
    if I_S.shape[0] < 20:
        print('I_S (optimal inventory levels): ', I_S)
    optimal_cost = sim.evaluate_cost(I_S=I_S, eval_num=100)  
    print('optimal cost of RNN-based method: %.3e' % optimal_cost)


def run_rnn_spanning_tree_optimization(data_type, temp_path, nodes_num):
    network_name_dict={10:"bom_kodak.pkl", #a simple example of Kodak digital camera supply chain
                       10000:"bom_spanning_tree_from_real_case_10000-2.pkl", #Larger Spanning Trees with 10000 nodes
                       50000:"bom_spanning_tree_from_real_case_50000.pkl"} #Larger Spanning Trees with 50000 nodes
    delivery_cycle_pkl_dict={10:'delivery_cycle-10nodes-2021-12-17 04-33.pkl',
                             10000:'delivery_cycle-10000nodes-2021-12-21 09-46-2.pkl'
                             ,50000:"delivery_cycle-50000nodes-2021-12-22 03-35.pkl"}
    step_size_dict={10:3.8e-2, 10000:2.53e-6, 50000:1e-5}
    regula_para_dict={10:1.2e4, 10000:1.46e6, 50000:7e5}
    stop_thresh_dict={10:2e-4, 10000:1e-4, 50000:1.11e-4}
    stop_thresh_ratio_dict={10:0.7, 10000:0.48, 50000:0.4}
    step_size_ratio_dict={10:0.08, 10000:0.026, 50000:0.0015}
    step_bound_dict={10:None, 10000:[[4,0.04,-4,-0.04], [1.5, 0.015, -1.5, -0.015]],
                     50000:[[3, 0.04, -3, -0.04], [2.5, 0.025, -2.5, -0.025]]}
    I_S0_K_dict={10:10, 10000:0.01, 50000:0.01}

    sim = simulation.Simulation(data_type=data_type, duration=100, data_path=temp_path,
                                network_name=network_name_dict[nodes_num],  # test_bom_spanning_tree50000.pkl
                                delivery_cycle=delivery_cycle_pkl_dict[nodes_num],
                                penalty_factor=2.0)  # "temp_delivery_cycle-10000-20210423.pkl"
    opt = simu_opt.SimOpt(data_path=temp_path, rep_num=10, step_size=step_size_dict[nodes_num],
                          regula_para=regula_para_dict[nodes_num],
                          stop_thresh=stop_thresh_dict[nodes_num], positive_flag=True, cost_f=sim.evaluate_cost,
                          grad_f=sim.evaluate_cost_gradient, step_bound=step_bound_dict[nodes_num],
                          stop_thresh_ratio=stop_thresh_ratio_dict[nodes_num],
                          step_size_ratio=step_size_ratio_dict[nodes_num], decay_mode=2)
    run_rnn_optimization(data_type=data_type, nodes_num=nodes_num, sim=sim, opt=opt, I_S0_K=I_S0_K_dict[nodes_num])


if __name__ == "__main__":
    data_path = "./data"
    data_type = np.float32
    nodes_num_list = [10]  # 可以根据需要添加更多节点数量，如[10, 10000, 50000]
    for n in nodes_num_list:
        run_rnn_spanning_tree_optimization(data_type, data_path, n)