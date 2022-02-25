import itertools
import os
import subprocess
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def make(simd="avx2", Olevel="-O3"):
    os.chdir("./iso3dfd-st7/")
    try:
        os.system(f"make clean")
    except Exception as e:
        print(e)
        pass
    os.system(f"make build simd={simd} Olevel={Olevel}")
    os.chdir("..")


def run(simd, Olevel, n1, n2, n3, num_thread, iteration, b1, b2, b3):
    basename = 'iso3dfd_dev13_cpu'
    exec_name = basename + '_'+str(simd) + '_'+str(Olevel)+'.exe'
    # filename = os.listdir("./iso3dfd-st7/bin/")[0]
    # print(filename)
    # print(n1, n2, num_thread, iteration, b1, b2, b3)
    p = subprocess.Popen([
        f"./iso3dfd-st7/bin/{exec_name}",
        str(n1),
        str(n2),
        str(n3),
        str(num_thread),
        str(iteration),
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


class AntColony():

    def __init__(self, alpha, beta, rho, Q, nb_ant, levels):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.nb_ant = nb_ant

        self.levels = levels
        self.graph = self.__init_graph()

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
        """ Choose the path of an ant """
        path = [("init", "init")]
        for _ in range(len(self.levels)-1):
            items_view = self.graph[path[-1]].items()
            neighbors = [a for (a, _) in items_view]
            neighbors_idx = np.arange(len(neighbors))
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
            for i in range(len(pathes[0]-1)):  # Reward best ant
                self.graph[path[i]][path[i+1]]['tau'] += extra_phero * \
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
        pathes = []
        performances = []
        l_results = []
        for _ in range(self.nb_ant):
            path = self.pick_path()
            # print(path)
            # make(path[1][1], path[2][1])
            results = run(path[1][1], path[2][1], n1=128, n2=128,
                          n3=128, iteration=10, **dict(path[3:]))
            performances.append(results[0])
            l_results.append((results, path))

        pathes = [path for _, path in sorted(
            zip(performances, pathes), key=lambda pair: pair[0])]
        l_results.sort(key=lambda x: x[0][0])
        # print(l_results)
        for element in l_results:
            print('Time to execute: %f. \nPath: %s' %
                  (element[0][0], str([item[1] for item in element[1]])))
        self.update_tau(pathes, method='basic')
        # print([(path,)])


alpha = 0.5
beta = 0
rho = 0.2
Q = 1
nb_ant = 5


block_min = 1
block_max = 256
block_size = 64


levels_exec = [("init", {"init"}),
               ("simd", {"avx", "avx2", 'avx512', 'sse'}),
               ("Olevel", {"-O2", "-O3", "-Ofast"}),
               ("num_thread", set([32])),
               #    ("num_thread", set([2**j for j in range(0, 6)])),
               ("b1", set(np.delete(np.arange(block_min-1, block_max+1, block_size), 0))),
               ("b2", set(np.delete(np.arange(block_min-1, block_max+1, block_size), 0))),
               ("b3", set(np.delete(np.arange(block_min-1, block_max+1, block_size), 0)))

               ]


# print(levels[3])

ant_colony = AntColony(alpha, beta, rho, Q, nb_ant, levels_exec)
# ant_colony.plot_graph()

epoch = 3
for k in range(epoch):
    print("EPOCH: %i" % k)
    ant_colony.epoch()
