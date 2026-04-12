import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rnnisa.model import simulation, simu_opt


DATA_PATH = "./data"
NETWORK_NAME = "bom_kodak_dual.pkl"
DELIVERY_CYCLE = "delivery_cycle-10nodes-2021-12-17 04-33.pkl"
NODES_NUM = 10
DURATION = 100
PENALTY_FACTOR = 2.0
REP_NUM = 10
DATA_TYPE = np.float64
I_S0_K = 20


def _make_init_points(sim):
    i_sr0 = DATA_TYPE(I_S0_K) * np.ones((1, NODES_NUM), dtype=DATA_TYPE)
    i_sr0[0, sim.get_demand_set()] = 40

    i_se0 = DATA_TYPE(0) * np.ones((1, NODES_NUM), dtype=DATA_TYPE)
    raw_node_indices = np.nonzero(sim.get_raw_nodes()[0])[0]
    i_se0[0, raw_node_indices] = I_S0_K
    return i_sr0, i_se0


class TrackingSimOpt(simu_opt.SimOpt):
    def __init__(self, *args, tracker=None, **kwargs):
        self._tracker = tracker if tracker is not None else {}
        super().__init__(*args, **kwargs)

    def FISTA(self, *args, **kwargs):
        self._tracker["phase"] = "fista"
        start = time.time()
        out = super().FISTA(*args, **kwargs)
        self._tracker["fista_time_sec"] = time.time() - start
        return out

    def SGD(self, *args, **kwargs):
        self._tracker["phase"] = "sgd"
        start = time.time()
        out = super().SGD(*args, **kwargs)
        self._tracker["sgd_time_sec"] = time.time() - start
        return out


def run_one(config, eval_num):
    simu_opt._print_opt_info = lambda *args, **kwargs: None
    simu_opt.print_run_time = lambda *args, **kwargs: None
    simu_opt.my_dump = lambda *args, **kwargs: None

    sim = simulation.Simulation(
        data_type=DATA_TYPE,
        duration=DURATION,
        data_path=DATA_PATH,
        network_name=NETWORK_NAME,
        delivery_cycle=DELIVERY_CYCLE,
        penalty_factor=PENALTY_FACTOR,
    )
    tracker = {"phase": None, "fista_iters": 0, "sgd_epochs": 0}

    def cost_f(i_sr, i_se, eval_count):
        return sim.evaluate_cost(I_Sr=i_sr, I_Se=i_se, eval_num=eval_count)

    def grad_f(i_sr, i_se, eval_count):
        if tracker["phase"] == "fista":
            tracker["fista_iters"] += 1
        elif tracker["phase"] == "sgd":
            tracker["sgd_epochs"] += 1
        return sim.evaluate_cost_gradient(I_Sr=i_sr, I_Se=i_se, eval_num=eval_count)

    opt = TrackingSimOpt(
        data_path=DATA_PATH,
        rep_num=REP_NUM,
        step_size=config["step_size"],
        step_size_e=config["step_size_e"],
        regula_para=config["regula_para"],
        stop_thresh=config["stop_thresh"],
        positive_flag=True,
        cost_f=cost_f,
        grad_f=grad_f,
        raw_nodes=sim.raw_node,
        step_bound=config["step_bound"],
        step_size_ratio=config["step_size_ratio"],
        stop_thresh_ratio=config["stop_thresh_ratio"],
        decay_mode=2,
        tracker=tracker,
    )

    i_sr0, i_se0 = _make_init_points(sim)
    start = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, i_sr, i_se = opt.two_stage_procedure(i_sr0, i_se0)
    train_time_sec = time.time() - start

    sim.reset_seed()
    with contextlib.redirect_stdout(io.StringIO()):
        final_cost = sim.evaluate_cost(I_Sr=i_sr, I_Se=i_se, eval_num=eval_num)

    return {
        "config": config,
        "train_time_sec": train_time_sec,
        "fista_iters": tracker["fista_iters"],
        "sgd_epochs": tracker["sgd_epochs"],
        "fista_time_sec": tracker.get("fista_time_sec"),
        "sgd_time_sec": tracker.get("sgd_time_sec"),
        "final_cost": float(final_cost),
        "nonzero_sr": int(np.count_nonzero(i_sr)),
        "i_sr": np.asarray(i_sr).round(6).tolist(),
        "i_se": np.asarray(i_se).round(6).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-size", type=float, default=1.5e-1)
    parser.add_argument("--step-size-e", type=float, default=1.5e-2)
    parser.add_argument("--regula-para", type=float, default=1.2e3)
    parser.add_argument("--stop-thresh", type=float, default=1e-3)
    parser.add_argument("--stop-thresh-ratio", type=float, default=1.0)
    parser.add_argument("--step-size-ratio", type=float, default=0.3)
    parser.add_argument("--i-s0-k", type=float, default=20)
    parser.add_argument("--eval-num", type=int, default=30)
    args = parser.parse_args()

    global I_S0_K
    I_S0_K = args.i_s0_k

    config = {
        "step_size": args.step_size,
        "step_size_e": args.step_size_e,
        "regula_para": args.regula_para,
        "stop_thresh": args.stop_thresh,
        "stop_thresh_ratio": args.stop_thresh_ratio,
        "step_size_ratio": args.step_size_ratio,
        "step_bound": None,
        "i_s0_k": args.i_s0_k,
    }
    result = run_one(config, eval_num=args.eval_num)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
