import itertools
import pickle
from mpl_toolkits.mplot3d import Axes3D
import getopt
import sys
import os
import subprocess
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import make_all
from tqdm import tqdm

from config import N1, N2, N3, NUM_ITER

import mpi4py
from mpi4py import MPI


def run(simd, Olevel, num_thread, b1, b2, b3):
    basename = 'iso3dfd_dev13_cpu'
    exec_name = basename + '_'+str(simd) + '_'+str(Olevel)+'.exe'
    p = subprocess.Popen([
        f"./iso3dfd-st7/bin/{exec_name}",
        N1,
        N2,
        N3,
        str(num_thread),
        NUM_ITER,
        str(b1),
        str(b2),
        str(b3)
    ],
        stdout=subprocess.PIPE)
    p.wait()
    outputs = p.communicate()[0].decode("utf-8").split("\n")
    time = float(outputs[-4].split(" ")[-2])
    throughput = float(outputs[-3].split(" ")[-2])
    flops = float(outputs[-2].split(" ")[-2])
    return time, throughput, flops


def save_results(lines, type='best'):
    """ Saves the reusults in a .txt file"""
    # print(lines)
    counter = 0
    filename = "Results_2_{}.txt"
    Path("./Results_2").mkdir(parents=True, exist_ok=True)
    # print('./Results/' + filename.format(counter))
    while os.path.isfile('./Results_2/' + filename.format(counter)):
        counter += 1
    filename = './Results_2/' + filename.format(counter)
    # print(filename)
    filename_pickle = filename[:-4] + '_pickle.plk'
    with open(filename_pickle, 'wb') as file_data:
        pickle.dump(lines, file_data)

    with open(filename, 'w') as f:
        for epoch, result_epoch in enumerate(lines):
            f.write('\n %--------')
            f.write('\n Epoch: %s\n' % epoch)
            if type == 'best':
                best = list(result_epoch)[0]
                f.write('\n Path: %s' % str([item[1]
                                             for item in best[1]]))
                f.write('\n Result: %s' % str(best[0]))
            else:
                for ant in result_epoch:
                    # f.write('Time to execute: %.3f || Throughput: %.3f || Flops: %.3f' % (
                    #     ant[0][0], ant[0][1], ant[0][2]))
                    f.write('\n Path: %s' % str([item[1]
                            for item in ant[1]]))
                    f.write('\n Result:' + str(ant[0]))
                    # f.write('\n Path: %s' % str([item[1]
                    #         for item in ant[1]]))
                    f.write('\n')


class AntColony():

    def __init__(self, alpha, beta, rho, Q, nb_ant, levels, method, local_search_method):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.nb_ant = nb_ant

        self.levels = levels
        self.graph = self.__init_graph()
        self.method = method
        self.local_search_method = local_search_method

    def __init_graph(self):
        """ Initialize the graph """
        graph = nx.DiGraph()
        # Initialisation des noeuds
        for level, choices in self.levels:
            for choice in choices:
                graph.add_node((level, choice), level=level)
        # Initialisation des liens
        for (name_i, choices_i), (name_j, choices_j) in zip(self.levels, self.levels[1:]):
            for choice_i in choices_i:
                for choice_j in choices_j:
                    graph.add_edge((name_i, choice_i),
                                   (name_j, choice_j), tau=1, nu=1)
        # print(graph)
        return graph

    def plot_graph(self):
        """ Show the graph """
        pos = nx.nx_pydot.graphviz_layout(self.graph, prog="dot", root="init")
        edges, tau = zip(*nx.get_edge_attributes(self.graph, 'tau').items())
        nx.draw(self.graph, pos, node_size=10, edgelist=edges,
                edge_color=tau, edge_cmap=plt.cm.plasma)
        plt.show()

    def pick_path(self):
        """
        Choose the path of an ant

        Return:
        path : list = [(name_lvl1, choice_lvl1), ...]
        """
        # Start from initial node
        path = [("init", "init")]
        for _ in range(len(self.levels)-1):
            items_view = self.graph[path[-1]].items()
            # List next nodes
            neighbors = [a for (a, _) in items_view]
            neighbors_idx = np.arange(len(neighbors))

            # Choose a node according to weights
            tau = np.array([e["tau"]
                            for (_, e) in items_view], dtype=np.float32)
            nu = np.array([e["nu"] for (_, e) in items_view], dtype=np.float32)
            weights = (tau**self.alpha) * (nu**self.beta)
            weights /= np.sum(weights)
            path.append(neighbors[np.random.choice(neighbors_idx, p=weights)])
        return path

    def update_tau(self, pathes):
        """ Updates the amount of pheromone on each edge based on the method choosen """
        # Basic Algorithm
        if self.method == 'basic':
            # Evaporation:
            for origin, destiny in self.graph.edges(data=False):
                self.graph[origin][destiny]['tau'] = (
                    1-self.rho)*self.graph[origin][destiny]['tau']
            for path in pathes:
                for i in range(len(path)-1):
                    self.graph[path[i]][path[i+1]]['tau'] += self.Q / \
                        (1/self.graph[path[i]][path[i+1]]['nu'])

        # ASrank
        if self.method == 'asrank':
            n_to_update = 5
            # Evaporation:
            for origin, destiny, edge in self.graph.edges(data=True):
                self.graph[origin][destiny]['tau'] = (
                    1-self.rho)*self.graph[origin][destiny]['tau']

            # Adding pheromone weighted by path's rank
            for path in pathes[:n_to_update]:
                weight = 1
                for i in range(len(path)-1):
                    self.graph[path[i]][path[i+1]]['tau'] += weight*self.Q / \
                        (1/self.graph[path[i]][path[i+1]]['nu'])
                weight += -1/n_to_update

        # Elitist Ant System (Elistist AS)
        if self.method == 'elitist':
            # Evaporation:
            for origin, destiny, edge in self.graph.edges(data=True):
                self.graph[origin][destiny]['tau'] = (
                    1-self.rho)*self.graph[origin][destiny]['tau']

            extra_phero = 1  # If extra_phero = 1, the ant adds 2 times more than other ants
            for i in range(len(pathes[0])-1):  # Reward best ant
                self.graph[pathes[0][i]][pathes[0][i+1]]['tau'] += extra_phero * \
                    self.Q/(1/self.graph[pathes[0][i]][pathes[0][i+1]]['nu'])

            # Adding pheromone
            for path in pathes:
                for i in range(len(path)-1):
                    self.graph[path[i]][path[i+1]]['tau'] += self.Q / \
                        (1/self.graph[path[i]][path[i+1]]['nu'])

        # MMAS
        if self.method == 'mmas':
            tau_min = 0.1
            tau_max = 10.0
            n_to_update = 5
            # Evaporation
            for origin, destiny in self.graph.edges(data=False):
                update = (1-self.rho)*self.graph[origin][destiny]['tau']
                self.graph[origin][destiny]['tau'] = max(update, tau_min)

            # Adding pheromone weighted by path's rank
            for path in pathes[:n_to_update]:
                weight = 1
                for i in range(len(path)-1):
                    self.graph[path[i]][path[i+1]]['tau'] += weight*self.Q / \
                        (1/self.graph[path[i]][path[i+1]]['nu'])
                    self.graph[path[i]][path[i+1]]['tau'] = min(self.graph[path[i]][path[i+1]]['tau'], tau_max)
                weight -= 1/n_to_update

    def epoch(self):
        # pathes and perfornamces of all ants of that generation
        pathes = []
        performances = []

        # Routine for each ant
        for _ in range(self.nb_ant):
            # 1- Pick a path
            path = self.pick_path()

            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[process 0] ant: ', _, ' path: ', path, file=sys.stderr)
            # 2- Do a local search
            path, cost = self.local_search_method.search_fn(self.levels, path)

            pathes.append(path)
            performances.append(cost)

        return pathes, performances


class Communication():
    def __init__(self):
        # MPI information extraction
        self.comm = MPI.COMM_WORLD
        self.NbP = self.comm.Get_size()
        self.Me = self.comm.Get_rank()


class IndependentColonies(Communication):
    def initial_communication(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, ant_colony, pathes, performances):
        pathes = [path for _, path in sorted(
            zip(performances, pathes), key=lambda pair: pair[0])]
        performances.sort()
        ant_colony.update_tau(pathes, method='mmas')
        return pathes, performances

    def last_communication(self, best_path, best_cost):
        print(f"Best cost of colony {self.Me}: {best_cost}")
        self.comm.Barrier()
        for i in range(self.NbP):
            cost = self.comm.bcast(best_cost, root=i)
            path = self.comm.bcast(best_path, root=i)
            if cost < best_cost:
                best_cost = cost
                best_path = path
        return best_path, best_cost


class ExchangeAll(Communication):
    def initial_communication(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, ant_colony, pathes, performances):
        # Communicate results
        pathes = self.comm.allreduce(pathes)
        performances = self.comm.allreduce(performances)
        # Sort all results
        pathes = [path for _, path in sorted(
            zip(performances, pathes), key=lambda pair: pair[0])]
        performances.sort()
        print(pathes)
        daskjfhasdjf
        # Update pheromones
        ant_colony.update_tau(pathes)
        return pathes, performances

    def last_communication(self, best_path, best_cost):
        self.comm.Barrier()
        for i in range(self.NbP):
            cost = self.comm.bcast(best_cost, root=i)
            path = self.comm.bcast(best_path, root=i)
            if cost < best_cost:
                best_cost = cost
                best_path = path
        return best_path, best_cost


def getBlockSizes(pathes):
    b1 = []
    b2 = []
    b3 = []
    for path in pathes:
        path = dict(path)
        b1.append(path['b1'])
        b2.append(path['b2'])
        b3.append(path['b3'])

    return b1, b2, b3


def main():

    # Compile at each machine:

    argv = sys.argv[1:]
    if len(argv) < 2:
        str_error = '[error] : incorrect number of parameters\n \
            Usage: python antcolony_mpi.py -m 0'
        raise Exception(str_error)
    try:
        opts, _args = getopt.getopt(argv,"m")
    except getopt.GetoptError as e:
        print('[error] : ', e)


    debug = True
    make = int(_args[0])

    # Parameters
    from localsearch import Identity
    #Parameters
    alpha = 1
    beta = 0
    rho = 0.6
    Q = 1
    nb_ant = 2
    nb_epochs = 1

    block_min = 1
    block_max = 32
    block_size = 16

    levels = [("init", ["init"]),
              ("simd", ["avx", "avx2", "avx512", "sse"]),
              ("Olevel", ["-O2", "-O3", "-Ofast"]),
              ("num_thread", [31,32]),
              ("b1", list(np.delete(np.arange(block_min-1, block_max+1, block_size), 0))),
              ("b2", list(np.delete(np.arange(block_min-1, block_max+1, block_size), 0))),
              ("b3", list(np.delete(np.arange(block_min-1, block_max+1, block_size), 0)))
              ]

    method = "mmas"
    local_search_method = Identity()
    communication = ExchangeAll()

    ant_colony = AntColony(alpha, beta, rho, Q, nb_ant, levels, method, local_search_method)

    best_path = None
    best_cost = np.inf

    communication.initial_communication()
    to_save = []

    b_pathes = []
    b_costs = []

    if communication.Me == 0:
        pbar = tqdm(total=nb_epochs, desc="Epoch Me: "+str(communication.Me))  # Loading bar

    for _ in range(nb_epochs):
        communication.on_epoch_begin()
        pathes, performances = ant_colony.epoch()
        communication.comm.Barrier()
        pathes, performances = communication.on_epoch_end(
            ant_colony, pathes, performances)
        if performances[0] < best_cost:
            best_path = pathes[0]
            best_cost = performances[0]

        if communication.Me == 0:
            save_results([zip(performances, pathes)], 'all')
            pbar.update(1)

    if communication.Me == 0:    
        pbar.close() 


    #best_path, best_cost = communication.last_communication(
        #best_path, best_cost)

    #if communication.Me == 0:
        #print("Best path: ", best_path)
        #print("Best cost: ", best_cost)


        #b1, b2, b3 = getBlockSizes(b_pathes)
        #fig = plt.figure()
        #ax = fig.add_subplot(1, 1, 1, projection='3d')
        #ax.scatter(b1, b2, b3, c=b_costs)
        #plt.savefig("3Dresult.png")

        #fig = plt.figure()
        #plt.title("Ant Colony - Solutions over the epochs")
        #plt.xlabel("Epoch")
        #plt.ylabel("Elapsed time")
        #plt.plot(np.arange(nb_epochs), b_costs)
        #plt.savefig("result.png")
if __name__ == "__main__":
    main()
