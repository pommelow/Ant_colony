import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from time import time
from antcolony_mpi import AntColony, IndependentColonies
from localsearch import Identity, GreedySearch, SimulatedAnnealing, RandomizedTabu

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=1, help="Control the pheromons")
parser.add_argument("--beta", type=float, default=1, help="Not used")
parser.add_argument("--rho", type=float, default=0.5, help="Control the pheromons evaporation")
parser.add_argument("--Q", type=float, default=1, help="Control the reward of the best ants")
parser.add_argument("--nb_ant", type=int, default=25, help="Number of ants in each generation")
parser.add_argument("--method", type=str, default="mmas", choices=["basic", "asrank", "elitist", "mmas"], help="")
parser.add_argument("--local-search", type=str, default="identity", choices=["identidy", "greedy", "simulated_annealing", "tabu"], help="identity means no local search")
parser.add_argument("--max_time", type=float, default=1800, help="maximum execution time of the ant colony")

# method parameters
parser.add_argument("--nb-to-update", type=int, default=12, help="nb of ants to update in asrank ans mmas methods")
parser.add_argument("--tau_min", type=float, default=0.5, help="tau min in mmas")
parser.add_argument("--tau_max", type=float, default=10, help="tau max in mmas")

# local search parameters
parser.add_argument("--time_local", type=float, default=600, help="time before local search")
parser.add_argument("--kmax", type=float, default=3, help="max iteration for each local search method")
parser.add_argument("--t0", type=float, default=50, help="initial temperature in simulated annealing")
parser.add_argument("--t-decay", type=float, default=0.98, help="temperature decay in simulated annealing")
parser.add_argument("--t_min", type=float, default=0.01, help="minimum temperature in simulated annealing")
parser.add_argument("--tabu-size", type=int, default=5, help="size of memory in tabu search")

# distribution parameters


args = parser.parse_args()
print(args)

block_min = 16
block_max = 256

levels = [("init", ["init"]),
            ("n1", list(range(256, 1024+1, 32))),
            ("n2", list(range(256, 1024+1, 32))),
            ("n3", list(range(256, 1024+1, 32))),
            ("simd", ["avx", "avx2", "avx512"]),
            ("Olevel", ["-O2", "-O3", "-Ofast"]),
            ("num_thread", [16]),
            ("b1", list(range(128, 512+1, 16))),
            ("b2", list(range(8, 64+1, 1))),
            ("b3", list(range(128, 256+1, 1)))
            ]

if args.local_search == "identity":
    local_search_method = Identity()
elif args.local_search == "greedy":
    local_search_method = GreedySearch(args.kmax)
elif args.local_search == "simulated_annealing":
    local_search_method = SimulatedAnnealing(args.kmax, args.t0, lambda t: max(args.t_min, args.t_decay * t))
elif args.local_search == "tabu":
    local_search_method = RandomizedTabu(args.tabu_size, args.k_max, args.k_max, args.t0, lambda t: max(args.t_min, args.t_decay * t))

communication = IndependentColonies()


method_args = {"nb_to_update": args.nb_to_update, "tau_min": args.tau_min, "tau_max": args.tau_max}

for _ in range(1):

    ant_colony = AntColony(args.alpha, args.beta, args.rho, args.Q, args.nb_ant, levels, args.method, Identity(), **method_args)

    best_cost = np.inf
    best_path = []
    history = {"best_cost": [], "time": []}

    top = time()

    communication.initial_communication()
    while time()-top < args.max_time:
        if time()-top > args.time_local:
            ant_colony.local_search_method = local_search_method

        print("Plot graph")
        ant_colony.plot_graph()
    
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

    folder_name = f"./Results/{args.alpha}_{args.beta}_{args.rho}_{args.Q}_{args.nb_ant}_{args.method}_identity_simAnn{args.kmax}_{args.t0}_{args.t_decay}"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    n = len(os.listdir(folder_name))
    with open(folder_name + f"/{n:>03}", "wb") as file:
        pickle.dump(history, file)
