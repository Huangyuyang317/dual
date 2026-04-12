import os
import numpy as np
from random import seed, normalvariate
from multiprocessing import Pool, cpu_count
from scipy.sparse import diags
from warnings import filterwarnings

filterwarnings('ignore')

CORE_NUM = cpu_count()


class Simulation():
    def __init__(self, data_type, duration, data_path, network_name, delivery_cycle=0, penalty_factor=10.0):
        self.__duration = duration
        self.__data_type = data_type
        self._prepare_data(data_path, network_name, penalty_factor, data_type, delivery_cycle)
        self.raw_node = self.__raw_material_node
        self.__seed_num = 0
        self._print_info()

    def _prepare_data(self, data_path, network_name, penalty_factor, data_type, delivery_cycle):
        import networkx as nx
        from scipy.sparse import eye
        from rnnisa.utils.tool_function import my_load

        def count_layer(B):
            B.eliminate_zeros()
            B = B.tocsr()
            B = B.astype(self.__data_type)
            temp = eye(B.shape[0], dtype=self.__data_type)
            temp = temp.tocsr()
            
            maxlayer = 0
            for i in range(B.shape[0]):
                temp = B * temp
                temp.eliminate_zeros()
                if temp.nnz == 0:
                    maxlayer = i
                    break
            return maxlayer + 1

        G = my_load(os.path.join(data_path, network_name))

        if type(G) == list:
            G = G[0]
        self.__B = nx.adjacency_matrix(G, weight='weight')
        self.__stage_num = count_layer(self.__B)
        self.__nodes_num = self.__B.shape[0]
        
        self.__c_fast = np.array(list(nx.get_node_attributes(G, 'cost_fast').values()), dtype=data_type).reshape(1, -1)
        self.__c_slow = np.array(list(nx.get_node_attributes(G, 'cost_slow').values()), dtype=data_type).reshape(1, -1)
        self.__lt_fast = np.array(list(nx.get_node_attributes(G, 'lt_fast').values()), dtype=int)
        self.__lt_slow = np.array(list(nx.get_node_attributes(G, 'lt_slow').values()), dtype=int)


        self.__delta_lt = self.__lt_slow - self.__lt_fast

        self.__hold_coef = np.array(list(nx.get_node_attributes(G, 'holdcost').values()), dtype=data_type).reshape(1, -1)

        self.__D_mean = np.zeros((self.__duration, self.__nodes_num))  
        self.__std = np.zeros_like(self.__D_mean)
        in_degree_values = np.array([v for k, v in G.in_degree()])
        demand_node = np.where(in_degree_values == 0)[0]
        i = 0
        for nd in list(G.nodes()):
            if i in demand_node:
                self.__D_mean[range(self.__duration), i] = G.nodes[nd]['mean']
                self.__std[range(self.__duration), i] = G.nodes[nd]['std']
            i += 1
        self.__ts_fast = np.zeros((self.__duration, self.__nodes_num), dtype=int)
        self.__ts_slow = np.zeros((self.__duration, self.__nodes_num), dtype=int)
        for t in range(self.__duration):
            self.__ts_fast[t, :] = np.minimum(t + self.__lt_fast, self.__duration)
            self.__ts_slow[t, :] = np.minimum(t + self.__lt_slow, self.__duration)
        self.__demand_set = demand_node

        self.__penalty_coef = data_type(penalty_factor) * self.__hold_coef
        self.__B = self.__B.astype(data_type)
        self.__B_T = self.__B.T
        self.__B_T = self.__B_T.tocsr()
        E = eye(self.__nodes_num, dtype=data_type)
        self.__E = E.tocsr()
        E_B_T = (self.__E - self.__B).T
        self.__E_B_T = E_B_T.tocsr()
        self.__E_B_T.eliminate_zeros()

        self.__zero = data_type(0.0)
        self.__one_minus = data_type(-1.0)
        self.__one = data_type(1.0)
        if data_type == np.float32:
            self.__equal_tole = data_type(1e-5)
        else:
            self.__equal_tole = data_type(1e-11)

        out_degree_values = np.expand_dims(np.array([v for k, v in G.out_degree()]), axis=0)
        self.__raw_material_node = np.where(out_degree_values == 0, self.__one, self.__zero)

        self.__c_fast = self.__c_fast * self.__raw_material_node
        self.__c_slow = self.__c_slow * self.__raw_material_node

        mau_item = self.__one - self.__raw_material_node
        self.__mau_item_diag = diags(mau_item[0])

        idx_mau = np.nonzero(1 - self.__raw_material_node)[1]
        self.__B_indices_list = {i: self.__B.getrow(i).indices for i in
                                 idx_mau}

        if type(delivery_cycle) == int:
            delivery_cycles = delivery_cycle * np.ones(self.__nodes_num, dtype=int)
        else:
            delivery_cycles = my_load(os.path.join(data_path, delivery_cycle))

        self.__delivery_shift = np.zeros((self.__duration, self.__nodes_num), dtype=int)
        for t in range(self.__duration):
            self.__delivery_shift[t, self.__demand_set] = t - delivery_cycles[self.__demand_set]
        self.__delivery_shift = np.maximum(-1, self.__delivery_shift)

    def _print_info(self):
        print('Data Type:', self.__data_type)
        print('nodes number:', self.__nodes_num)

    def reset_seed(self):
        self.__seed_num = 0

    def cut_seed(self, num):
        self.__seed_num -= num

    def get_demand_set(self):
        return self.__demand_set

    def get_raw_nodes(self):
        return self.__raw_material_node

    def evaluate_cost(self, I_Sr, I_Se, eval_num=30):
        process_num = min(CORE_NUM, eval_num)
        if self.__nodes_num == 500000: process_num = min(process_num, 30)
        if self.__nodes_num == 100000: process_num = min(process_num, 50)

        I_S_list = [(I_Sr, I_Se, self.__duration, self.__nodes_num, self.__zero, self.__one, self.__one_minus, self.__stage_num,
             self.__data_type, self.__B_indices_list,
             self.__hold_coef, self.__penalty_coef, self.__c_slow, self.__c_fast, self.__raw_material_node,
             self.__B, self.__ts_slow, self.__ts_fast, self.__delta_lt, 
             self.__D_mean, self.__std, self.__demand_set, self.__delivery_shift, i + self.__seed_num) 
            for i in range(eval_num)]
        self.__seed_num += eval_num
        with Pool(process_num) as pool:
            result = pool.map(_simulate_only_parallel, I_S_list)

        cost = np.mean(result)

        return cost

    def evaluate_cost_gradient(self, I_Sr, I_Se, eval_num=30, mean_flag=True):
        process_num = min(CORE_NUM, eval_num)
        I_S_list = [(I_Sr, I_Se, self.__duration, self.__nodes_num, self.__zero, self.__one, self.__one_minus, self.__stage_num,
             self.__lt_slow, self.__lt_fast, self.__data_type, self.__B_indices_list, self.__equal_tole,
             self.__hold_coef, self.__penalty_coef, self.__c_slow, self.__c_fast, self.__mau_item_diag, self.__raw_material_node,
             self.__B, self.__B_T, self.__E_B_T, self.__ts_slow, self.__ts_fast, self.__delta_lt, 
             self.__D_mean, self.__std, self.__demand_set, self.__delivery_shift, i + self.__seed_num) 
            for i in range(eval_num)]
        self.__seed_num += eval_num
        with Pool(process_num) as pool:
            result = pool.map(_simulate_and_bp_parallel, I_S_list)
            
        result = list(zip(*result))
        cost_result = np.array(result[0])
        grad_result_r = np.squeeze(result[1])
        grad_result_e = np.squeeze(result[2])
        if mean_flag:
            cost_result = np.mean(cost_result)
            grad_result_r = np.mean(grad_result_r, axis=0,
                                  keepdims=True)
            grad_result_e = np.mean(grad_result_e, axis=0,
                                  keepdims=True)
        return cost_result, grad_result_r, grad_result_e

    # def evaluate_cost_gradient(self, I_Sr, I_Se, eval_num=30, mean_flag=True):
    #         process_num = min(CORE_NUM, eval_num)
    #         I_S_list = [(I_Sr, I_Se, self.__duration, self.__nodes_num, self.__zero, self.__one, self.__one_minus, self.__stage_num,
    #             self.__data_type, self.__B_indices_list,
    #             self.__hold_coef, self.__penalty_coef, self.__c_slow, self.__c_fast, self.__raw_material_node,
    #             self.__B, self.__ts_slow, self.__ts_fast, self.__delta_lt, 
    #             self.__D_mean, self.__std, self.__demand_set, self.__delivery_shift, i + self.__seed_num) 
    #             for i in range(eval_num)]
    #         self.__seed_num += eval_num
    #         with Pool(process_num) as pool:
    #             result = pool.map(_simulate_and_bp_tf, I_S_list)
                
    #         result = list(zip(*result))
    #         cost_result = np.array(result[0])
    #         grad_result_r = np.squeeze(result[1])
    #         grad_result_e = np.squeeze(result[2])
    #         if mean_flag:
    #             cost_result = np.mean(cost_result)
    #             grad_result_r = np.mean(grad_result_r, axis=0,
    #                                 keepdims=True)
    #             grad_result_e = np.mean(grad_result_e, axis=0,
    #                                 keepdims=True)
    #         return cost_result, grad_result_r, grad_result_e

    def get_standard_args(self, I_Sr, I_Se, seed_val):
            return (I_Sr, I_Se, self.__duration, self.__nodes_num, self.__zero, self.__one, self.__one_minus, self.__stage_num,
             self.__lt_slow, self.__lt_fast, self.__data_type, self.__B_indices_list, self.__equal_tole,
             self.__hold_coef, self.__penalty_coef, self.__c_slow, self.__c_fast, self.__mau_item_diag, self.__raw_material_node,
             self.__B, self.__B_T, self.__E_B_T, self.__ts_slow, self.__ts_fast, self.__delta_lt, 
             self.__D_mean, self.__std, self.__demand_set, self.__delivery_shift, seed_val)

def _generate_random_demand_parallel(duration, nodes_num, data_type, D_mean, std, demand_set, delivery_shift,
                                     zero, rand_seed):
    if rand_seed is not None:
        np.random.seed(rand_seed)
    D = np.zeros((duration, nodes_num), dtype=data_type)
    D_order = np.zeros((duration + 1, nodes_num), dtype=data_type)
    for t in range(duration):
        D_order[t, demand_set] = np.random.normal(D_mean[t, demand_set], std[t, demand_set])
        D[t, demand_set] = D_order[delivery_shift[t, demand_set], demand_set]

    D_order = np.maximum(zero, D_order)
    D = np.maximum(zero, D)
    return D, D_order


def _simulate_and_bp_parallel(args):
    (I_Sr, I_Se, duration, nodes_num, zero, one, one_minus, stage_num,
     lt_slow, lt_fast, data_type, B_indices_list, equal_tole,
     hold_coef, penalty_coef, c_slow, c_fast, mau_item_diag, raw_material_node,
     B, B_T, E_B_T, ts_slow, ts_fast, delta_lt, 
     D_mean, std, demand_set, delivery_shift, rand_seed) = args
    maximum = np.maximum
    minimum = np.minimum
    where = np.where
    nonzero = np.nonzero
    np_abs = np.abs
    zeros_like = np.zeros_like
    np_sum = np.sum
    np_multiply = np.multiply
    np_array = np.array

    D, D_order = _generate_random_demand_parallel(duration, nodes_num, data_type, D_mean, std, demand_set, delivery_shift,
                                                  zero, rand_seed)

    M_backlog = np.zeros((1, nodes_num), dtype=data_type)
    P = np.zeros((duration + 1, nodes_num), dtype=data_type)
    Pr = np.zeros((duration + 1, nodes_num), dtype=data_type)
    D_backlog = zeros_like(M_backlog)
    I_t = I_Sr + zero
    I_Pr = I_Sr + zero
    I_Pe = I_Sr + zero
    cost = zero
    
    d_It_d_Yt, d_Dback_d_Yt = [], []
    d_Or_d_IPr_stack = [[] for _ in range(duration)]
    d_Oe_d_IPe_stack = [[] for _ in range(duration)]
    d_M_d_man_o = [zeros_like(I_Sr) for _ in range(duration)]
    d_M_d_r_r = [{} for _ in range(duration)]
    d_r_r_d_I, d_r_r_d_r_n = [], []

    for t in range(duration):
        I_Pr = I_Pr - D_order[t, :]
        I_Pe = I_Pe - D_order[t, :]
        I_Pe = I_Pe + Pr[t, :]

        Oe_t = maximum(zero, I_Se - I_Pe) * raw_material_node
        flag_e = where(I_Se - I_Pe > 0, one_minus, zero) * raw_material_node
        d_Oe_d_IPe_stack[t].insert(0, diags((flag_e)[0])) 
        
        Or_t = maximum(zero, I_Sr - (I_Pr + Oe_t))
        flag_r = where(I_Sr - (I_Pr + Oe_t) > 0, one_minus, zero)
        d_Or_d_IPr_stack[t].insert(0, diags((flag_r)[0]))

        for _ in range(stage_num - 1):
            temp_internal_D = (Or_t + Oe_t) * B
            Oe_t = maximum(zero, I_Se - (I_Pe - temp_internal_D)) * raw_material_node
            flag_e = where(I_Se - (I_Pe - temp_internal_D) > 0, one_minus, zero) * raw_material_node
            d_Oe_d_IPe_stack[t].insert(0, diags((flag_e)[0]))
            
            Or_t = maximum(zero, I_Sr - (I_Pr - temp_internal_D + Oe_t))
            flag_r = where(I_Sr - (I_Pr - temp_internal_D + Oe_t) > 0, one_minus, zero)
            d_Or_d_IPr_stack[t].insert(0, diags((flag_r)[0]))

        valid_t = np.minimum(t + delta_lt, duration)
        Pr[valid_t, range(nodes_num)] = Or_t[0, :]

        total_O_t = Or_t + Oe_t
        internal_D_t = total_O_t * B
        I_Pr = I_Pr + total_O_t - internal_D_t
        I_Pe = (I_Pe + Oe_t - internal_D_t) * raw_material_node

        temp_I_val = I_t - D_backlog - D[t] + P[t]
        I_t = maximum(zero, temp_I_val)
        d_It_d_Yt.append(diags(where(temp_I_val > 0, one, zero)[0]))
        D_backlog = -minimum(zero, temp_I_val)
        d_Dback_d_Yt.append(diags(where(temp_I_val <= 0, one_minus, zero)[0]))

        purchase_order_e = Oe_t * raw_material_node
        purchase_order_r = Or_t * raw_material_node
        mau_o = (Oe_t + Or_t - purchase_order_e - purchase_order_r + M_backlog)*(1 - raw_material_node)
        idx_purch = nonzero((Oe_t+Or_t) * raw_material_node)[1] 
        idx_mau = nonzero(mau_o * (one -raw_material_node))[1]

        P[ts_fast[t, idx_purch], idx_purch] += purchase_order_e[0, idx_purch]
        P[ts_slow[t, idx_purch], idx_purch] += purchase_order_r[0, idx_purch]

        res_needed = mau_o * B
        temp_rate = I_t / res_needed
        temp_rate[res_needed == 0] = one
        res_rate = minimum(one, temp_rate)
        
        flag_res = where(temp_rate < 1, one, zero)
        inv_res = one / res_needed
        inv_res[res_needed == 0] = one
        d_r_r_d_I.append(diags(np_multiply(flag_res, inv_res)[0]))
        d_r_r_d_r_n.append(diags(np_multiply(flag_res, -np_multiply(temp_rate, inv_res))[0]))

        M_actual = zeros_like(M_backlog)
        min_rate = np_array([res_rate[0, B_indices_list[idx]].min() if len(B_indices_list[idx]) > 0 else 1.0 for idx in idx_mau])
        M_actual[0, idx_mau] = min_rate * mau_o[0, idx_mau]
        
        col2 = [B_indices_list[idx_mau[i]][np_abs(res_rate[0, B_indices_list[idx_mau[i]]] - min_rate[i]) < equal_tole] 
             for i in range(len(idx_mau))]
        d_M_d_r_r[t] = {idx_mau[i]: (data_type(1.0 / len(col2[i])) * mau_o[0, idx_mau[i]], col2[i]) for i in range(len(idx_mau)) 
            if min_rate[i] > 0}
        d_M_d_man_o[t][0, idx_mau] = min_rate

        P[ts_slow[t, idx_mau], idx_mau] += M_actual[0, idx_mau]
        I_t = I_t - M_actual * B 
        M_backlog = mau_o - M_actual
        cost += np_sum(np_multiply(I_t, hold_coef)) + np_sum(np_multiply(D_backlog, penalty_coef)) + np_sum(np_multiply(Or_t, c_slow)) + np_sum(np_multiply(Oe_t, c_fast))

    d_Sr, d_Se = zeros_like(I_Sr), zeros_like(I_Se)
    d_It, d_Dback = hold_coef + zero, penalty_coef + zero
    d_IPr, d_IPe = zeros_like(I_Sr), zeros_like(I_Se)
    d_Mt_backlog = zeros_like(I_Sr)
    d_Or = np.zeros((duration, 1, nodes_num), dtype=data_type)
    d_Oe = np.zeros((duration, 1, nodes_num), dtype=data_type)

    d_Pr_buf = np.zeros_like(d_Or)
    d_P_buf = np.zeros_like(d_Or)

    for t in range(duration - 1, -1, -1):
        d_Mact = - d_It * B_T
        d_Mq = d_Mact - d_Mt_backlog + d_P_buf[t]
        d_mau_o = (d_Mt_backlog + np_multiply(d_Mq, d_M_d_man_o[t]))*(1 - raw_material_node)
        d_res_r = zeros_like(I_Sr)
        for idx in d_M_d_r_r[t]:
            val, cols = d_M_d_r_r[t][idx]
            d_res_r[0, cols] += val * d_Mq[0, idx]
            
        d_It = d_It + d_res_r * d_r_r_d_I[t]
        d_res_n = d_res_r * d_r_r_d_r_n[t]
        d_mau_o = (d_mau_o + d_res_n * B_T)*(1 - raw_material_node)
        d_Or[t] += d_mau_o * mau_item_diag + c_slow * raw_material_node
        d_Oe[t] += c_fast * raw_material_node
        d_Yt = d_It * d_It_d_Yt[t] + d_Dback * d_Dback_d_Yt[t]
        d_Or[t] += d_IPr * E_B_T - d_IPe * B_T
        d_Oe[t] += (d_IPr + d_IPe) * E_B_T * raw_material_node

        d_temp_Or = d_Or[t] + zero
        d_temp_Oe = d_Oe[t] + zero
        
        for i in range(stage_num - 1):
            d_t_IPr = d_temp_Or * d_Or_d_IPr_stack[t][i]
            d_Sr -= d_t_IPr
            d_IPr += d_t_IPr
            d_temp_Oe += d_t_IPr 
            d_t_IPe = d_temp_Oe * d_Oe_d_IPe_stack[t][i] * raw_material_node
            d_Se -= d_t_IPe
            d_IPe += d_t_IPe
            d_temp_Or = - (d_t_IPr + d_t_IPe) * B_T
            d_temp_Oe = d_temp_Or * raw_material_node

        temp_d_IPr = d_temp_Or * d_Or_d_IPr_stack[t][stage_num - 1]
        d_Sr -= temp_d_IPr
        d_IPr += temp_d_IPr
        d_temp_Oe += temp_d_IPr 

        temp_d_IPe = d_temp_Oe *d_Oe_d_IPe_stack[t][stage_num - 1]
        d_Se -= temp_d_IPe
        d_IPe += temp_d_IPe
        if t > 0:
            d_Mt_backlog = d_mau_o + zero
            d_It = d_Yt + hold_coef
            d_Dback = -d_Yt + penalty_coef
            
            idx_slow = nonzero(P[t])[0]
            valid_slow_idx = idx_slow[t - lt_slow[idx_slow] >= 0]
            if len(valid_slow_idx) > 0:
                valid_slow_t = t - lt_slow[valid_slow_idx]
                d_Or[valid_slow_t, 0, valid_slow_idx] += d_Yt[0, valid_slow_idx]

            idx_fast = nonzero(P[t] * raw_material_node[0])[0]
            valid_fast_idx = idx_fast[t - lt_fast[idx_fast] >= 0]
            if len(valid_fast_idx) > 0:
                valid_fast_t = t - lt_fast[valid_fast_idx]
                d_Oe[valid_fast_t, 0, valid_fast_idx] += d_Yt[0, valid_fast_idx]

            idx_m = nonzero(P[t] * mau_item_diag.diagonal())[0]
            valid_m_idx = idx_m[t - lt_slow[idx_m] >= 0]
            if len(valid_m_idx) > 0:
                valid_m_t = t - lt_slow[valid_m_idx]
                d_P_buf[valid_m_t, 0, valid_m_idx] += d_Yt[0, valid_m_idx]

            idx_pr = nonzero((Pr[t] * raw_material_node[0]> 0))[0]
            valid_pr_idx = idx_pr[t - delta_lt[idx_pr] >= 0]
            if len(valid_pr_idx) > 0:
                valid_pr_t = t - delta_lt[valid_pr_idx]
                d_Or[valid_pr_t, 0, valid_pr_idx] += d_IPe[0, valid_pr_idx]
    # I_t, I_Pr and I_Pe are initialized from I_Sr before the forward loop.
    # Their remaining adjoints after the reverse sweep must be accumulated
    # back into the starting-inventory gradient.
    d_Sr += d_It + d_IPr + d_IPe
    d_Se = d_Se * raw_material_node

    return cost, d_Sr, d_Se


def _simulate_only_parallel(args):
    (I_Sr, I_Se, duration, nodes_num, zero, one, one_minus, stage_num,
    data_type, B_indices_list,
     hold_coef, penalty_coef, c_slow, c_fast, raw_material_node,
     B, ts_slow, ts_fast, delta_lt, 
     D_mean, std, demand_set, delivery_shift, rand_seed) = args

    maximum = np.maximum
    minimum = np.minimum
    where = np.where
    nonzero = np.nonzero
    np_abs = np.abs
    zeros_like = np.zeros_like
    np_sum = np.sum
    np_multiply = np.multiply
    np_array = np.array

    D, D_order = _generate_random_demand_parallel(duration, nodes_num, data_type, D_mean, std, demand_set, delivery_shift,
                                                  zero, rand_seed)

    M_backlog = np.zeros((1, nodes_num), dtype=data_type)
    P = np.zeros((duration + 1, nodes_num), dtype=data_type)
    Pr = np.zeros((duration + 1, nodes_num), dtype=data_type)
    D_backlog = zeros_like(M_backlog)
    I_t = I_Sr + zero
    I_Pr = I_Sr + zero
    I_Pe = I_Sr + zero
    cost = zero
    

    for t in range(duration):
        I_Pr = I_Pr - D_order[t, :]
        I_Pe = I_Pe - D_order[t, :]
        I_Pe = I_Pe + Pr[t, :]

        Oe_t = maximum(zero, I_Se - I_Pe) * raw_material_node
        Or_t = maximum(zero, I_Sr - (I_Pr + Oe_t))

        for _ in range(stage_num - 1):
            temp_internal_D = (Or_t + Oe_t) *B
            Oe_t = maximum(zero, I_Se - (I_Pe - temp_internal_D)) * raw_material_node
            Or_t = maximum(zero, I_Sr - (I_Pr - temp_internal_D + Oe_t))

        valid_t = np.minimum(t + delta_lt, duration)
        Pr[valid_t, range(nodes_num)] = Or_t[0, :]

        total_O_t = Or_t + Oe_t
        internal_D_t = total_O_t * B
        I_Pr = I_Pr + total_O_t - internal_D_t
        I_Pe = I_Pe + Oe_t - internal_D_t

        temp_I_val = I_t - D_backlog - D[t] + P[t]
        I_t = maximum(zero, temp_I_val)
        D_backlog = -minimum(zero, temp_I_val)

        purchase_order_e = Oe_t * raw_material_node
        purchase_order_r = Or_t * raw_material_node
        mau_o = Oe_t + Or_t - purchase_order_e - purchase_order_r + M_backlog
        idx_purch = nonzero((Oe_t+Or_t) * raw_material_node)[1] 
        idx_mau = nonzero(mau_o * (one -raw_material_node))[1]

        P[ts_fast[t, idx_purch], idx_purch] += purchase_order_e[0, idx_purch]
        P[ts_slow[t, idx_purch], idx_purch] += purchase_order_r[0, idx_purch]

        res_needed = mau_o * B
        temp_rate = I_t / res_needed
        temp_rate[res_needed == 0] = one
        res_rate = minimum(one, temp_rate)

        M_actual = zeros_like(M_backlog)
        min_rate = np_array([res_rate[0, B_indices_list[idx]].min() if len(B_indices_list[idx]) > 0 else 1.0 for idx in idx_mau])
        M_actual[0, idx_mau] = min_rate * mau_o[0, idx_mau]

        P[ts_slow[t, idx_mau], idx_mau] += M_actual[0, idx_mau]
        I_t = I_t - M_actual * B 
        M_backlog = mau_o - M_actual

        cost += np_sum(np_multiply(I_t, hold_coef)) + np_sum(np_multiply(D_backlog, penalty_coef)) + np_sum(np_multiply(Or_t, c_slow)) + np_sum(np_multiply(Oe_t, c_fast))
    return cost

def _simulate_and_bp_tf(args):
    import tensorflow as tf
    (I_Sr, I_Se, duration, nodes_num, zero, one, one_minus, stage_num,
     lt_slow, lt_fast, data_type, B_indices_list, equal_tolerance,
     hold_coef, penalty_coef, c_slow, c_fast, mau_item_diag, raw_material_node,
     B, B_T, E_B_T, ts_slow, ts_fast, delta_lt, 
     D_mean, std, demand_set, delivery_shift, rand_seed) = args
    forward_args_template = (
        duration, nodes_num, zero, one, one_minus, stage_num,
        data_type, B_indices_list, hold_coef, penalty_coef, c_slow, c_fast,
        raw_material_node, B, ts_slow, ts_fast, delta_lt,
        D_mean, std, demand_set, delivery_shift, rand_seed
    )
    backward_args_template = (
        duration, nodes_num, zero, one, one_minus, stage_num,
        lt_slow, lt_fast, data_type, B_indices_list, equal_tolerance,
        hold_coef, penalty_coef, c_slow, c_fast, mau_item_diag, raw_material_node,
        B, B_T, E_B_T, ts_slow, ts_fast, delta_lt,
        D_mean, std, demand_set, delivery_shift, rand_seed
    )

    @tf.custom_gradient
    def tf_cost_with_manual_grad(tf_I_Sr, tf_I_Se):
        sr_np = tf_I_Sr.numpy()
        se_np = tf_I_Se.numpy()
        cost_np = _simulate_only_parallel((sr_np, se_np) + forward_args_template)
        cost_tensor = tf.convert_to_tensor(cost_np, dtype=data_type)

        def grad(dy):
            _, grad_r_np, grad_e_np = _simulate_and_bp_parallel((sr_np, se_np) + backward_args_template)
            grad_r_tensor = tf.convert_to_tensor(grad_r_np, dtype=data_type)
            grad_e_tensor = tf.convert_to_tensor(grad_e_np, dtype=data_type)
            return dy * grad_r_tensor, dy * grad_e_tensor

        return cost_tensor, grad

    tf_I_Sr = tf.constant(I_Sr, dtype=data_type)
    tf_I_Se = tf.constant(I_Se, dtype=data_type)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch([tf_I_Sr, tf_I_Se])
        cost = tf_cost_with_manual_grad(tf_I_Sr, tf_I_Se)
    gradients = tape.gradient(cost, [tf_I_Sr, tf_I_Se])
    grad_Sr = gradients[0].numpy() if gradients[0] is not None else np.zeros_like(I_Sr)
    grad_Se = gradients[1].numpy() if gradients[1] is not None else np.zeros_like(I_Se)
    cost = cost.numpy()
    _print_cost_grad_info(cost, grad_Sr, grad_Se)
    return cost, grad_Sr, grad_Se


def _print_cost_grad_info(cost, grad_Sr, grad_Se):
    print('-------------------------------------------')
    print('total_cost: ', cost)
    
    if grad_Sr.shape[1] > 666:
        print(f'gradient of item 666 (Sr): {grad_Sr[0, 666]:.6f}')
        print(f'gradient of item 666 (Se): {grad_Se[0, 666]:.6f}')
    cost_change_Sr = np.sum(grad_Sr) 
    cost_change_Se = np.sum(grad_Se)
    
    print(f'Sr sum grad (total change if all Sr+1): {cost_change_Sr:.4f}')
    print(f'Se sum grad (total change if all Se+1): {cost_change_Se:.4f}')
    print(f'Total predicted cost change: {cost_change_Sr + cost_change_Se:.4f}')
    print('-------------------------------------------')
