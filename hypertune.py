import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from antcolony_mpi import AntColony, IndependentColonies
from localsearch import Identity

# alpha=0.5
beta=1
rho=0.1
Q=1
nb_ant=15

block_min = 16
block_max = 256
block_size = 16

levels = [("init", ["init"]),
            ("simd", ["avx", "avx2", "avx512", "sse"]),
            ("Olevel", ["-O2", "-O3", "-Ofast"]),
            ("num_thread", list([2**j for j in range(0, 6)])),
            ("b1", list(np.arange(block_min, block_max+1, block_size))),
            ("b2", list(np.arange(block_min, block_max+1, block_size))),
            ("b3", list(np.arange(block_min, block_max+1, block_size)))
            ]

method = "mmas"
local_search_method = Identity()

communication = IndependentColonies()



alpha_l = [0.1, 0.1, 0.1]  # , 0.5, 1, 1.25, 1.5, 2]
max_time = 180

plt.figure()

for alpha in tqdm(alpha_l):

    ant_colony = AntColony(alpha, beta, rho, Q, nb_ant, levels, method, local_search_method)

    best_cost = np.inf
    best_path = []
    history = {"best_cost": [], "time": []}

    top = time()

    communication.initial_communication()
    while time()-top < max_time:
        communication.on_epoch_begin()
        pathes, performances = ant_colony.epoch()
        pathes, performances = communication.on_epoch_end(ant_colony, pathes, performances)
        
        if performances[0] < best_cost:
            best_path = pathes[0]
            best_cost = performances[0]
            history["best_cost"].append(best_cost)
            history["time"].append(time() - top)
    
    best_path, best_cost = communication.last_communication(best_path, best_cost)
    history["best_cost"].append(best_cost)
    history["time"].append(time() - top)

    plt.plot(history["time"], history["best_cost"], label=f"a = {alpha}")

plt.xlabel("Time")
plt.ylabel("Best cost")
plt.legend()
plt.show()
