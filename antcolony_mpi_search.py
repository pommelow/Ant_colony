import itertools
import os
import subprocess
from localsearch import GreedySearch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
import copy

from config import N1, N2, N3, NUM_ITER

import mpi4py
from mpi4py import MPI

def make(simd="avx2", Olevel="-O3"):
    os.chdir("./iso3dfd-st7/")
    try:
        subprocess.run(["make", "clean"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    except Exception as e:
        print(e)
        pass
    subprocess.run(["make", "build", f"simd={simd}",f" Olevel={Olevel} "],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

    os.chdir("..")


def run(simd, Olevel, num_thread, b1, b2, b3):
    basename = 'iso3dfd_dev13_cpu'
    exec_name = basename + '_'+str(simd) + '_'+str(Olevel)+'.exe'
    # filename = os.listdir("./iso3dfd-st7/bin/")[0]
    # print(filename)
    # print(n1, n2, num_thread, iteration, b1, b2, b3)
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


def save_results(lines):
    """ Saves the reusults in a .txt file"""
    # print(lines)
    counter = 0
    filename = "Results{}.txt"
    Path("./Results").mkdir(parents=True, exist_ok=True)
    print('./Results/' + filename.format(counter))
    while os.path.isfile('./Results/' + filename.format(counter)):
        counter += 1
    filename = './Results/' + filename.format(counter)
    print(filename)

    with open(filename, 'w') as f:
        for epoch, result_epoch in enumerate(lines):
            f.write('\n Epoch: %s\n' % epoch)
            for ant in result_epoch:
                f.write('Time to execute: %.3f || Throughput: %.3f || Flops: %.3f' % (
                    ant[0][0], ant[0][1], ant[0][2]))
                f.write('\n Path: %s' % str([item[1]
                        for item in ant[1]]))
                f.write('\n %---')


class AntColony():

    def __init__(self, alpha, beta, rho, Q, nb_ant, levels, local_search_method):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.nb_ant = nb_ant

        self.levels = levels
        self.graph = self.__init_graph()
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

    def update_tau(self, pathes, method='basic'):
        """ Updates the amount of pheromone on each edge based on the method choosen """
        # Basic Algorithm
        print('update_tau: ', pathes)
        if method == 'basic':
            # Evaporation:
            for origin, destiny in self.graph.edges(data=False):
                self.graph[origin][destiny]['tau'] = (
                    1-self.rho)*self.graph[origin][destiny]['tau']
            for path in pathes:
                for i in range(len(path)-1):
                    self.graph[path[i]][path[i+1]]['tau'] += self.Q / \
                        (1/self.graph[path[i]][path[i+1]]['nu'])

        # ASrank
        if method == 'asrank':
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
        if method == 'elitist':
            # Evaporation:
            for origin, destiny, edge in self.graph.edges(data=True):
                self.graph[origin][destiny]['tau'] = (
                    1-self.rho)*self.graph[origin][destiny]['tau']

            extra_phero = 1  # If extra_phero = 1, the ant adds 2 times more than other ants
            for i in range(len(pathes[0])-1):  # Reward best ant
                self.graph[pathes[0][i]][pathes[0][i+1]]['tau'] += extra_phero * \
                    self.Q/(1/self.graph[path[i]][path[i+1]]['nu'])

            # Adding pheromone
            for path in pathes:
                for i in range(len(path)-1):
                    self.graph[path[i]][path[i+1]]['tau'] += self.Q / \
                        (1/self.graph[path[i]][path[i+1]]['nu'])

        # MMAS
        if method == 'mmas':
            tau_min = 0.1
            tau_max = 10.0
            # Evaporation
            for origin, destiny in self.graph.edges(data=False):
                update = (1-self.rho)*self.graph[origin][destiny]['tau']
                self.graph[origin][destiny]['tau'] = max(update, tau_min)

            for i in range(len(pathes[0]-1)):  # Only best at adds pheromone
                increment = self.Q/(1/self.graph[path[i]][path[i+1]]['nu'])
                self.graph[path[i]][path[i+1]]['tau'] = min(
                    self.graph[path[i]][path[i+1]]['tau'] + increment, tau_max)

    def epoch(self):
        # pathes and perfornamces of all ants of that generation
        pathes = []
        performances = []

        # Routine for each ant
        for _ in range(self.nb_ant):
            # 1- Pick a path
            path = self.pick_path()
            # 2- Do a local search
            path, cost = self.local_search_method.search_fn(self.levels, path)

            pathes.append(path)
            performances.append(cost)

        # Sort pathes
        pathes = [path for _, path in sorted(zip(performances, pathes), key=lambda pair: pair[0])]
        performances.sort()


        print(f"Best path: {[e[1] for e in pathes[0]]}\nTime to execute: {performances[0]}")

        self.update_tau(pathes, method='basic')

        return pathes, performances


class IndependentColonies():
    def initial_communication(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, ant_colony, pathes, performances):
        ant_colony.update_tau(pathes, method='basic')



class Communication():
    def __init__(self):
        # MPI information extraction
        self.comm = MPI.COMM_WORLD
        self.NbP = self.comm.Get_size()
        self.Me = self.comm.Get_rank()


def divide_blocks(total_levels, block, process, NbP):
    total_levels = dict(total_levels)
    b_size = len(total_levels['b1']) // NbP
    return [val for val in total_levels[block][process*b_size:process*b_size + b_size]] 

class SearchSpaceDivision(Communication):
    
    def __init__(self, local_search=None, initial_levels=None, ant_colony_params=None, epochs=1):
        super().__init__()
        

        if not local_search:
            self.local_search_method = GreedySearch(kmax=1)
        else:
            self.local_search_method = local_search

        if not ant_colony_params:
            self.params = {
                'alpha': 0.5,
                'beta': 0,
                'rho': 0.2,
                'Q': 1,
                'nb_ant': 1,
                'block_min': 1,
                'block_max': 32,
                'block_size': 16
            }
        else:
            self.params = ant_colony_params

        if not initial_levels:
            block_min = self.params['block_min']
            block_max = self.params['block_max']
            block_size = self.params['block_size']

            self.total_levels = [["init", ["init"]],
            ["simd", ["avx", "avx2", "avx512", "sse"]],
            ["Olevel", ["-O2", "-O3", "-Ofast"]],
            ["num_thread", list([2**j for j in range(0, 6)])],
            ["b1", list(np.delete(np.arange(block_min-1, block_max+1, block_size), 0))],
            ["b2", list(np.delete(np.arange(block_min-1, block_max+1, block_size), 0))],
            ["b3", list(np.delete(np.arange(block_min-1, block_max+1, block_size), 0))]
            ]
        else: 
            self.total_levels = initial_levels
        


    def on_epoch_begin(self):
        self.levels = dict(copy.copy(self.total_levels))
        self.levels['b1'] = divide_blocks(self.total_levels, 'b1', self.Me, self.NbP)
        self.levels['b2'] = divide_blocks(self.total_levels ,'b2', self.Me, self.NbP)
        self.levels['b3'] = divide_blocks(self.total_levels, 'b3', self.Me, self.NbP)

        input_levels = []
        for keys, values in self.levels.items():
            input_levels.append((keys, values))

        print('Me: ', self.Me, ' levels: ', input_levels) 

        ant_colony_params = {
            'alpha': self.params['alpha'],
            'beta': self.params['beta'],
            'rho': self.params['rho'],
            'Q': self.params['Q'],
            'nb_ant': self.params['nb_ant'],
            'levels': input_levels,
            'local_search_method': self.local_search_method
        }

        return AntColony(**ant_colony_params)

    def on_epoch_end(self, ant_colony, pathes, performances):

        print('on_epoch_end pathes: ', pathes)
        pathes = self.comm.allreduce(pathes)
        print('all_reduced pathes: ', pathes)
        performances = self.comm.allreduce(performances)

        pathes = [path for _, path in sorted(zip(performances, pathes), key=lambda pair: pair[0])]
        performances.sort()

        print('pathes: ', pathes)
        ant_colony.update_tau(pathes, method='mmas')


        self.comm.Barrier()
        for i in range(self.NbP):
            cost = self.comm.bcast(best_cost, root=i)
            if cost < best_cost:
                best_cost = cost
                best = i

        self.total_levels[-3][1] = divide_blocks(self.total_levels, 'b1', best, self.NbP)
        self.total_levels[-2][1] = divide_blocks(self.total_levels ,'b2', best, self.NbP)
        self.total_levels[-1][1] = divide_blocks(self.total_levels, 'b3', best, self.NbP)

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

def main():

    parallel_model = SearchSpaceDivision()


    best_cost = np.inf 
    nb_epochs = 2
    for _ in range(nb_epochs):

        antColony = parallel_model.on_epoch_begin() 

        epochs = 2
        pathes = []
        for __ in range(epochs):
            path, performances = antColony.epoch()
            pathes.append(path)

        parallel_model.comm.Barrier()
        pathes, performances = parallel_model.on_epoch_end(antColony, pathes, performances)
        
        if performances[0] < best_cost:
            best_path = pathes[0]
            best_cost = performances[0]
    
    best_path, best_cost = parallel_model.last_communication(best_path, best_cost)

    if parallel_model.Me == 0:
        print("Best path: ", best_path)
        print("Best cost: ", best_cost)

if __name__ == "__main__":
   main() 