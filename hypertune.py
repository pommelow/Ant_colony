import enum
from antcolony import *
from localsearch import Identity
from itertools import product

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--t0", type=float)
parser.add_argument("--kmax", type=int)
args = parser.parse_args()

alpha=1
beta=1
rho=0.1
Q=1
nb_ant=10
method="elitist"

block_min = 16
block_max = 256

levels = [("init", ["init"]),
            ("n1", list(range(256, 1024+1, 32))),
            ("n2", list(range(256, 1024+1, 32))),
            ("n3", list(range(256, 1024+1, 32))),
            ("simd", ["avx", "avx2", "avx512"]),
            ("Olevel", ["-O2", "-O3", "-Ofast"]),
            ("num_thread", [16]),
            ("b1", [0]),
            ("b2", list(range(block_min, block_max+1, 1))),
            ("b3", list(range(block_min, block_max+1, 1)))
            ]

alpha_l=np.arange(5,55,5)/10
beta_l=[0]
rho_l=np.arange(0,10)/10
Q_l=[1]
nb_ant_l=np.arange(5,55,5)
methods=["basic","asrank","elitist","mmas"]

def create_l(list_of_param=[alpha_l,beta_l,rho_l,Q_l,nb_ant_l,methods]):
    products = itertools.product(*list_of_param)
    return list(products)

def hypertune(parameter_l):
    res=[]
    res_path=[]

    for parameter in parameter_l :
        alpha, beta, rho, Q, nb_ant,method=parameter
        print(f"Hyperparameters : alpha={alpha}, beta={beta}, rho={rho}, Q={Q}, nb_ant={nb_ant}, method={method}\n")

        epoch_max=10
        epoch=0
        #local_search_method = GreedySearch(kmax=1)
        local_search_method = Identity()
        ant_colony=AntColony(alpha, beta, rho, Q, nb_ant, levels, method,local_search_method)

        best_time=1000000
        current_time=np.inf
        best_path=[]
        current_path=[]

        while (abs(best_time-current_time)>0.1) and (epoch<=epoch_max):

            if best_time>current_time:
                best_time=current_time
                best_path=current_path

            epoch+=1
            pathes,performances=ant_colony.epoch()

            current_time=performances[0]
            current_path=[e[1] for e in pathes[0]]
        
        res.append(best_time)
        res_path.append(best_path)

    fig, ax = plt.subplots()
    ax.scatter(np.arange(len(res)),res)
    ax.set_ylabel("time")
    for i,text in enumerate(res_path):
        plt.annotate(res_path,(parameter_l[i],res[i]))
    plt.savefig("hypertune.png")
    return res,res_path





list_of_param=[[alpha],beta_l,[rho],[Q],[nb_ant],[method]]
hypertune(create_l(list_of_param))
