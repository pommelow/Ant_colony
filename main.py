import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from time import time
from antcolony_mpi import AntColony, ExchangeAll, IndependentColonies, folder_results, save_results
from localsearch import Identity, GreedySearch, SimulatedAnnealing, RandomizedTabu
from tqdm import tqdm

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=1,
                    help="Control the pheromons")
parser.add_argument("--beta", type=float, default=1, help="Not used")
parser.add_argument("--rho", type=float, default=0.1,
                    help="Control the pheromons evaporation")
parser.add_argument("--Q", type=float, default=1,
                    help="Control the reward of the best ants")
parser.add_argument("--nb_ant", type=int, default=25,
                    help="Number of ants in each generation")
parser.add_argument("--method", type=str, default="mmas",
                    choices=["basic", "asrank", "elitist", "mmas"], help="")
parser.add_argument("--local-search", type=str, default="identity", choices=[
                    "identidy", "greedy", "simulated_annealing", "tabu"], help="identity means no local search")
parser.add_argument("--max_time", type=float, default=1800,
                    help="maximum execution time of the ant colony")

# method parameters
parser.add_argument("--nb-to-update", type=int, default=12,
                    help="nb of ants to update in asrank ans mmas methods")
parser.add_argument("--tau_min", type=float,
                    default=0.5, help="tau min in mmas")
parser.add_argument("--tau_max", type=float,
                    default=10, help="tau max in mmas")

# local search parameters
parser.add_argument("--time_local", type=float, default=600,
                    help="time before local search")
parser.add_argument("--kmax", type=float, default=3,
                    help="max iteration for each local search method")
parser.add_argument("--t0", type=float, default=50,
                    help="initial temperature in simulated annealing")
parser.add_argument("--t-decay", type=float, default=0.98,
                    help="temperature decay in simulated annealing")
parser.add_argument("--t_min", type=float, default=0.01,
                    help="minimum temperature in simulated annealing")
parser.add_argument("--tabu-size", type=int, default=5,
                    help="size of memory in tabu search")

# distribution parameters
parser.add_argument("--distribution", type=str, default="independent", choices=[
                    "independent", "exchange_all"], help="how process communicate with each other")


args = parser.parse_args()
print(args)

block_min = 16
block_max = 256

# Graph of the the Ant colony
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

# levels = [("init", ["init"]),
#           ("n1", [512]),
#           ("n2", [512]),
#           ("n3", [1024]),
#           ("simd", ["avx", "avx2", "avx512"]),
#           ("Olevel", ["-O2", "-O3", "-Ofast"]),
#           ("num_thread", [16]),
#           ("b1", list(range(128, 512+1, 16))),
#           ("b2", list(range(8, 64+1, 1))),
#           ("b3", list(range(128, 256+1, 1)))
#           ]

# choose the local search method
if args.local_search == "identity":
    local_search_method = Identity()
elif args.local_search == "greedy":
    local_search_method = GreedySearch(args.kmax)
elif args.local_search == "simulated_annealing":
    local_search_method = SimulatedAnnealing(
        args.kmax, args.t0, lambda t: max(args.t_min, args.t_decay * t))
elif args.local_search == "tabu":
    local_search_method = RandomizedTabu(
        args.tabu_size, args.k_max, args.k_max, args.t0, lambda t: max(args.t_min, args.t_decay * t))

if args.distribution == "independent":
    communication = IndependentColonies()
if args.distribution == "exchange_all":
    communication = ExchangeAll()


method_args = {"nb_to_update": args.nb_to_update,
               "tau_min": args.tau_min, "tau_max": args.tau_max}

# Launch _ colony x times to reduce variance in results

for _ in range(1):

    # creation of the colony
    ant_colony = AntColony(args.alpha, args.beta, args.rho, args.Q,
                           args.nb_ant, levels, args.method, Identity(), **method_args)

    best_cost = np.inf
    best_path = []
    history = {"best_cost": [], "time": []}

    top = time()

    # distribution on the colony on several nodes

    communication.initial_communication()

    # Path to save files
    if communication.Me == 0:
        path_dir = folder_results()

    same_solution_counter = 0

    print('='*20)
    print('Me: ', communication.Me, 'hostname: ', os.uname()[1])

    print('Running with the following parameters: ', args)
    print('='*20)
    if communication.Me == 0:
        # Loading bar
        pbar = tqdm(total=args.max_time,
                    desc="Epoch Me: "+str(communication.Me))

    to_save = []
    # run the colony until max time
    while time()-top < args.max_time:
        top1 = time()
        # add local method after "time local"
        if time()-top > args.time_local:
            ant_colony.local_search_method = local_search_method

        communication.on_epoch_begin()

        # do 1 epoch
        pathes, performances = ant_colony.epoch()
        # share performances between different nodes
        communication.comm.Barrier()
        pathes, performances = communication.on_epoch_end(
            ant_colony, pathes, performances)

        # get the performance of the colony over time
        if performances[0] < best_cost:
            best_path = pathes[0]
            best_cost = performances[0]

        if communication.Me == 0:
            best_path_short = str([item[1] for item in best_path])
            print('Best path until epoch %s: %s' % (_, best_path_short))
            print('Best cost until epoch %s: %s' % (_, -best_cost))
            save_results(zip(pathes, performances), path_dir)
            to_save.append(zip(pathes, performances))
            pbar.update(int(time()-top1))

        if best_path == pathes[0]:
            same_solution_counter += 1
            if same_solution_counter >= 5:
                break
        else:
            same_solution_counter = 0

    best_path, best_cost = communication.last_communication(
        best_path, best_cost)

    if communication.Me == 0:
        dict_ants = {}
        for ant_index, ant in enumerate(to_save[0]):
            if ant_index == 0:
                headers_path = [item[0] for item in ant[0]]
                break
        dict_ants['Epoch'] = []
        dict_ants['Ant'] = []
        dict_ants['Performance'] = []
        for header in headers_path:
            dict_ants[str(header)] = []

        for epoch, lines in enumerate(to_save):
            for ant_index, ant in enumerate(lines):
                path_ant = [item[1] for item in ant[0]]
                perf_ant = abs(ant[1])
                dict_ants['Performance'].append(perf_ant)
                dict_ants['Epoch'].append(epoch)
                dict_ants['Ant'].append(ant_index)
                for header, parameter in zip(headers_path, path_ant):
                    dict_ants[str(header)].append(parameter)

        # Store data (serialize)
        with open(str(path_dir + "Results_final_pickle.pkl"), 'wb') as handle:
            pickle.dump(dict_ants, handle)

        pbar.close()
