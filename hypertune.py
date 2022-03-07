from antcolony import *

alpha=0.5
beta=0
rho=0.2
Q=1
nb_ant=10

block_min = 1
block_max = 256
block_size = 16

levels = [("init", ["init"]),
            ("simd", ["avx", "avx2", "avx512", "sse"]),
            ("Olevel", ["-O2", "-O3", "-Ofast"]),
            ("num_thread", list([2**j for j in range(0, 6)])),
            ("b1", list(np.delete(np.arange(block_min-1, block_max+1, block_size), 0))),
            ("b2", list(np.delete(np.arange(block_min-1, block_max+1, block_size), 0))),
            ("b3", list(np.delete(np.arange(block_min-1, block_max+1, block_size), 0)))
            ]

local_search_method = GreedySearch(kmax=1)




alpha_l=np.arange(5,55,5)/10
rho_l=np.arange(0,10)/10
nb_ant=np.arange(5,55,5)
methods=["basic","asrank","elitist","mmas"]

res=[]
res_path=[]

for alpha in alpha_l :

    epoch_max=10
    epoch=0
    ant_colony=AntColony(alpha, beta, rho, Q, nb_ant, levels, local_search_method)

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

    






plt.plot(alpha_l,res)
plt.xlabel("alpha")
plt.ylabel("time")
plt.show()

