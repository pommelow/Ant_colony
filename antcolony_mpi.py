import json
import pickle
import getopt
import sys
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import make_all
from tqdm import tqdm
import time

import mpi4py
from mpi4py import MPI



def folder_results():
    """ Saves the reusults in a .txt file"""
    # print(lines)
    counter = 0
    foldername = "Run{}"
    # Creates a ./Results folder
    Path("./Results").mkdir(parents=True, exist_ok=True)

    # Creates a ./Results/Run{} folder
    while os.path.isdir('./Results/' + foldername.format(counter)):
        counter += 1
    Path(str("./Results/"+foldername.format(counter))
         ).mkdir(parents=True, exist_ok=True)
    path_dir = "./Results/"+foldername.format(counter)+'/'

    return path_dir


def save_results(lines, path_dir):
    counter = 0
    filename = "Results{}.txt"
    filename_pickle = "Results{}_pickle.pkl"
    # Creates the .txt file in ./Results/Run{}
    while os.path.isfile(path_dir + filename.format(counter)):
        counter += 1
    filename = path_dir + filename.format(counter)
    filename_pickle = path_dir + filename_pickle.format(counter)


    global exec_time
    global last_time

    exec_time = time.time() - last_time
    dict_ants = {}
    for ant_index, ant in enumerate(lines):
        if ant_index == 0:
            headers_path = [item[0] for item in ant[0]]
            break

    dict_ants['Performance'] = []
    dict_ants['Time'] = []
    for header in headers_path:
        dict_ants[str(header)] = []

    # [(path1,perf1),...,(pathN,perfN)]
    with open(filename, 'w') as f:
        for ant_index, ant in enumerate(lines):
            path_ant = [item[1] for item in ant[0]]
            perf_ant = abs(ant[1])
            dict_ants['Performance'].append(perf_ant)
            dict_ants['Time'].append(exec_time)
            for header, parameter in zip(headers_path, path_ant):
                dict_ants[str(header)].append(parameter)

            f.write('\n Ant %s' % (ant_index))
            f.write('\n Path: %s' % (str(path_ant)))
            f.write('\n Throughput: %s' % (perf_ant))
            f.write('\n')
    # Store data (serialize)
    with open(filename_pickle, 'wb') as handle:
        pickle.dump(dict_ants, handle)


def check_size(b1, b2, b3):
    """ Check if the cache block size is greatter than the cache size (l3) """
    cache_size = 11264000
    size = b1*b2*b3*4
    return size > cache_size


class AntColony():

    def __init__(self, alpha, beta, rho, Q, nb_ant, levels, method, local_search_method, **kwargs):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.nb_ant = nb_ant

        self.levels = levels
        self.graph = self.__init_graph()
        self.method = method
        self.local_search_method = local_search_method

        self.kwargs = kwargs

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
        tau_min = min(tau)
        edges_, tau_ = [], []
        for e, t in zip(edges, tau):
            if t != tau_min:
                edges_.append(e)
                tau_.append(t)
        nx.draw(self.graph, pos, node_size=10, edgelist=edges_,
                edge_color=tau_, edge_cmap=plt.cm.OrRd)
        plt.show()

    def pick_path(self):
        """
        Choose the path of an ant

        Return:
        path : list = [(name_lvl1, choice_lvl1), ...]
        """
        size_ok = False
        while not size_ok:

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
                nu = np.array([e["nu"]
                              for (_, e) in items_view], dtype=np.float32)
                weights = (tau**self.alpha) * (nu**self.beta)
                weights /= np.sum(weights)
                path.append(
                    neighbors[np.random.choice(neighbors_idx, p=weights)])
            b1, b2, b3 = getBlockSizes([path])
            if not check_size(b1[0], b2[0], b3[0]):
                size_ok = True

        return path

    def update_tau(self, performances, pathes):
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
            n_to_update = self.kwargs["nb_to_update"]
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
        if self.method == 'mmas_v2':
            tau_min = self.kwargs["tau min"]
            tau_max = self.kwargs["tau max"]
            n_to_update = self.kwargs["n to update"]
            # Evaporation
            for origin, destiny in self.graph.edges(data=False):
                update = (1-self.rho)*self.graph[origin][destiny]['tau']
                self.graph[origin][destiny]['tau'] = max(update, tau_min)

            # Adding pheromone weighted by path's rank
            for path_idx in range(n_to_update):
                path = pathes[path_idx]
                weight = performances[path_idx]/np.array(performances[:n_to_update]).sum()

                for i in range(len(path)-1):
                    self.graph[path[i]][path[i+1]]['tau'] += weight*self.Q / \
                        (1/self.graph[path[i]][path[i+1]]['nu'])
                    self.graph[path[i]][path[i+1]]['tau'] = min(
                        self.graph[path[i]][path[i+1]]['tau'], tau_max)
                weight -= 1/n_to_update

        if self.method == 'mmas':
            tau_min = self.kwargs["tau_min"]
            tau_max = self.kwargs["tau_max"]
            n_to_update = self.kwargs["nb_to_update"]
            # Evaporation
            for origin, destiny in self.graph.edges(data=False):
                update = (1-self.rho)*self.graph[origin][destiny]['tau']
                self.graph[origin][destiny]['tau'] = max(update, tau_min)

            # Adding pheromone weighted by path's rank
            for path_idx in range(n_to_update):
                path = pathes[path_idx]
                weight = performances[path_idx]/np.array(performances[:n_to_update]).sum()

                for i in range(len(path)-1):
                    self.graph[path[i]][path[i+1]]['tau'] += weight*self.Q / \
                        (1/self.graph[path[i]][path[i+1]]['nu'])
                    self.graph[path[i]][path[i+1]]['tau'] = min(
                        self.graph[path[i]][path[i+1]]['tau'], tau_max)
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
                # print('[process 0] ant: ', _, ' path: ', path, file=sys.stderr)
                pass
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
        ant_colony.update_tau(performances, pathes)
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
        # Update pheromones
        ant_colony.update_tau(performances, pathes)
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
