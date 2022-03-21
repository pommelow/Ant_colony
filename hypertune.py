import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from time import time
from antcolony_mpi import AntColony, IndependentColonies
from localsearch import Identity, GreedySearch, SimulatedAnnealing

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--t0", type=float)
parser.add_argument("--kmax", type=int)
args = parser.parse_args()

alpha=1
beta=1
rho=0.1
Q=1
nb_ant=25

block_min = 16
block_max = 256

levels = [("init", ["init"]),
            ("n1", list(range(512, 4096+1, 128))),
            ("n2", list(range(512, 4096+1, 128))),
            ("n3", list(range(512, 4096+1, 128))),
            ("simd", ["avx", "avx2", "avx512"]),
            ("Olevel", ["-O2", "-O3", "-Ofast"]),
            ("num_thread", [16]),
            ("b1", [0]),
            ("b2", list(range(block_min, block_max+1, 1))),
            ("b3", list(range(block_min, block_max+1, 1)))
            ]

method = "mmas"
mmas_args = {"tau min": 0.05, "tau max": 10, "n to update": 12}
local_search_method = Identity()

communication = IndependentColonies()


max_time = 1200


for _ in range(1):

    ant_colony = AntColony(alpha, beta, rho, Q, nb_ant, levels, method, local_search_method, **mmas_args)

    best_cost = np.inf
    best_path = []
    history = {"best_cost": [], "time": []}

    top = time()

    communication.initial_communication()
    while time()-top < max_time:
        if time()-top > 600:
            ant_colony.local_search_method = SimulatedAnnealing(kmax=args.kmax, t0=args.t0)
        communication.on_epoch_begin()
        pathes, performances = ant_colony.epoch()
        pathes, performances = communication.on_epoch_end(ant_colony, pathes, performances)

        if performances[0] < best_cost:
            best_path = pathes[0]
            best_cost = performances[0]
            history["best_cost"].append(best_cost)
            history["time"].append(time() - top)

        print(f"Time: {time() - top:.1f}s\nBest cost: {best_cost}\nBest path:{best_path}")
    
    best_path, best_cost = communication.last_communication(best_path, best_cost)
    history["best_cost"].append(best_cost)
    history["time"].append(time() - top)

    folder_name = f"./Results/{alpha}_{beta}_{rho}_{Q}_{nb_ant}_{method}_identity_simAnn{args.kmax}_{args.t0}"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    n = len(os.listdir(folder_name))
    with open(folder_name + f"/{n:>03}", "wb") as file:
        pickle.dump(history, file)
