import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


import mpi4py
from mpi4py import MPI


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
        if self.method == 'mmas':
            tau_min = self.kwargs["tau_min"]
            tau_max = self.kwargs["tau_max"]
            n_to_update = self.kwargs["nb_to_update"]
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
        pathes = [path for _, path in sorted(zip(performances, pathes), key=lambda pair: pair[0])]
        performances.sort()
        ant_colony.update_tau(pathes)
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
        pathes = [path for _, path in sorted(zip(performances, pathes), key=lambda pair: pair[0])]
        performances.sort()
        # Update pheromones
        ant_colony.update_tau(pathes)
        return pathes, performances
    
    def last_communication(self, best_path, best_cost):
        for i in range(self.NbP):
            cost = self.comm.bcast(best_cost, root=i)
            path = self.comm.bcast(best_path, root=i)
            if cost < best_cost:
                best_cost = cost
                best_path = path
        return best_path, best_cost
