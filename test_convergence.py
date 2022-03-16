from cmath import tau
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from time import time
from antcolony_mpi import AntColony, IndependentColonies,run
from localsearch import Identity

def conv_colony():
    alpha=np.ramdom.uniform(0,10)
    Q=np.ramdom.uniform(0,10)
    beta=1
    rho=np.ramdom.uniform(0,10)/10
    nb_ant=int(np.ramdom.uniform(0,100))

    tau_min=0.1
    tau_max=10
    n_to_update=5
    block_min = 16
    block_max = 256
    block_size = 16

    levels = [("init", ["init"]),
            ("simd", ["avx", "avx2", "avx512", "sse"]),
            ("Olevel", ["-O2", "-O3", "-Ofast"]),
            ("num_thread", [31,32,63,64]),
            ("b1", list(np.arange(block_min, block_max+1, block_size))),
            ("b2", list(np.arange(block_min, block_max+1, block_size))),
            ("b3", list(np.arange(block_min, block_max+1, block_size)))
            ]

    method = "mmas"
    local_search_method = Identity()

    communication = IndependentColonies()


    max_time = 600

    plt.figure()

    for _ in range(1):

        kwargs={"tau_max":tau_max,"tau_min":tau_min,"n_to_update":n_to_update}

        ant_colony = AntColony(alpha, beta, rho, Q, nb_ant, levels, method, local_search_method,**kwargs)

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

        folder_name = f"./Results_conv/{alpha}_{beta}_{rho}_{Q}_{nb_ant}_{method}_identity"
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        n = len(os.listdir(folder_name))
        with open(folder_name + f"/{n:>03}", "wb") as file:
            pickle.dump(history, file)

def conv_prog(n):
    block_min = 16
    block_max = 256
    block_size = 16
    t=[]        
    simd=np.random.choice(["avx", "avx2", "avx512", "sse"])
    Olevel=np.random.choice(["-O2", "-O3", "-Ofast"])
    num_thread=np.random.choice([31,32,63,64])
    b1=np.random.choice(list(np.arange(block_min, block_max+1, block_size)))
    b2=np.random.choice(list(np.arange(block_min, block_max+1, block_size)))
    b3=np.random.choice(list(np.arange(block_min, block_max+1, block_size)))
    for i in range(n):

        t.append(run(simd, Olevel, num_thread, b1, b2, b3)[0])
    label=str([simd,Olevel,num_thread,b1,b2,b3])
    return(t,label)
    plt.scatter(np.arange(n),t,label=str([simd,Olevel,num_thread,b1,b2,b3]))
    plt.axhline(y = np.mean(t), color = 'r', linestyle = '-',label=f"mean={round(np.mean(t),3)},std={round(np.std(t),3)}")
    plt.legend()
    plt.ylabel("execution time")
    plt.ylim((0,1))
    plt.show()

if __name__=="__main__":
    times_mean=[]
    times_std=[]
    labels=[]
    for i in range(10):
        t,label=conv_prog(10)
        times_mean.append(round(np.mean(t)))
        times_std.append(round(np.std(t)))
        labels.append(labels)

    fig, ax1 = plt.subplots()
    ax1.scatter(labels,times_mean)
    ax1.set_ylabel("means")
    ax1.set_xlabel("hyperparameters")
    ax2 = ax1.twinx()
    ax1.ylim((0,1))
    ax2.scatter(labels,times_std)
    ax2.ylim((0,0.1))
    ax2.set_ylabel("std")
    
    fig.tight_layout()
    plt.show()


