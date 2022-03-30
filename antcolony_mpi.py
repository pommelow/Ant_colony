import itertools
import json
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
import time

from config import N1, N2, N3, NUM_ITER

import mpi4py
from mpi4py import MPI


def make(simd="avx2", Olevel="-O3"):
    os.chdir("./iso3dfd-st7/")
    try:
        subprocess.run(["make", "clean"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(e)
        pass
    subprocess.run(["make", "build", f"simd={simd}", f" Olevel={Olevel} "],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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


pickle_data = {'index': [], 'n1': [], 'n2': [], 'n3': [], 'simd': [], 'Olevel': [], 'num_thread': [], 'b1': [], 'b2': [], 'b3': [], 'throughput': [], 'time': []}

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
    # [(path1,perf1),...,(pathN,perfN)]
    with open(filename, 'w') as f:
        for ant_index, ant in enumerate(lines):
            path_ant = str([item[1] for item in ant[0]])
            perf_ant = abs(ant[1])
            f.write('\n Ant %s' % (ant_index))
            f.write('\n Path: %s' % (path_ant))
            f.write('\n Throughput: %s' % (perf_ant))
            f.write('\n')
            pickle_data['index'].append(ant_index)
            pickle_data['n1'].append(dict(ant[0])['n1'])
            pickle_data['n2'].append(dict(ant[0])['n2'])
            pickle_data['n3'].append(dict(ant[0])['n3'])
            pickle_data['simd'].append(dict(ant[0])['simd'])
            pickle_data['Olevel'].append(dict(ant[0])['Olevel'])
            pickle_data['num_thread'].append(dict(ant[0])['num_thread'])
            pickle_data['b1'].append(dict(ant[0])['b1'])
            pickle_data['b2'].append(dict(ant[0])['b2'])
            pickle_data['b3'].append(dict(ant[0])['b3'])
            pickle_data['throughput'].append(perf_ant)
            pickle_data['time'].append(exec_time)



def check_size(b1, b2, b3):
    cache_size = 11264000
    size = b1*b2*b3*4
    bigger = False
    if size > cache_size:
        bigger = True
    return bigger


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
        nx.draw(self.graph, pos, node_size=10, edgelist=edges,
                edge_color=tau, edge_cmap=plt.cm.plasma)
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

    def epoch(self):
        # pathes and perfornamces of all ants of that generation
        pathes = []
        performances = []

        # Routine for each ant
        for _ in range(self.nb_ant):
            # 1- Pick a path
            path = self.pick_path()

            if MPI.COMM_WORLD.Get_rank() == 0:
                pass
                # print('[process 0] ant: ', _, ' path: ', path, file=sys.stderr)
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
        ant_colony.update_tau(performances, pathes, method='mmas')
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

last_time = time.time()
exec_time = 0

def main(args):

    global exec_time
    exec_time = time.time()
    # Parameters
    from localsearch import Identity
    # Parameters
    block_min = args['block_min']
    block_max = args['block_max']
    block_size = args['block_size']

    levels = [("init", ["init"]),
              ("n1", [512]),
              ("n2", [512]),
              ("n3", [512]),
              ("simd", ["avx", "avx2", "avx512"]),
              ("Olevel", ["-O2", "-O3", "-Ofast"]),
              ("num_thread", [16]),
              ("b1", list(np.delete(np.arange(block_min-1, block_max+1, block_size), 0))),
              ("b2", list(np.delete(np.arange(block_min-1, block_max+1, block_size), 0))),
              ("b3", list(np.delete(np.arange(block_min-1, block_max+1, block_size), 0)))
              ]
    method = "mmas"
    mmas_args = {"tau min": 0.1, "tau max": 5, "n to update": 75}

    local_search_method = Identity()
    communication = ExchangeAll()

    alpha = args['alpha']

    ant_colony = AntColony(args['alpha'], args['beta'], args['rho'], args['Q'], args['nb_ant'],
                           levels, method, local_search_method, **mmas_args)

    best_path = None
    best_cost = np.inf

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
        pbar = tqdm(total=args['nb_epochs'],
                    desc="Epoch Me: "+str(communication.Me))

    for _ in range(args['nb_epochs']):

        communication.on_epoch_begin()
        pathes, performances = ant_colony.epoch()
        communication.comm.Barrier()
        pathes, performances = communication.on_epoch_end(
            ant_colony, pathes, performances)
        if performances[0] < best_cost:
            best_path = pathes[0]
            best_cost = performances[0]

        if communication.Me == 0:
            best_path_short = str([item[1] for item in best_path])
            print('Best path until epoch %s: %s' % (_, best_path_short))
            print('Best cost until epoch %s: %s' % (_, -best_cost))
            save_results(zip(pathes, performances), path_dir)
            pbar.update(1)

        if best_path == pathes[0]:
            same_solution_counter += 1
            if same_solution_counter >= 5:
                break
        else:
            same_solution_counter = 0

    if communication.Me == 0:

        global pickle_data
        filename_pickle = "Results_pickle.plk"
        filename_pickle = path_dir + filename_pickle
        with open(filename_pickle, 'wb') as f:
            pickle.dump(pickle_data, f)
        pbar.close()


if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) < 2:
        str_error = '[error] : incorrect number of parameters\n \
            Usage: python antcolony_mpi.py -c training_config.json (or --config training_config.json)'
        raise Exception(str_error)
    try:
        opts, _args = getopt.getopt(argv, "c:", ['config='])
    except getopt.GetoptError as e:
        print('[error] : ', e)

    for opt, arg in opts:
        if opt in ['-c', '--config']:
            config_path = arg

    with open(config_path, 'r') as config_file:
        args = json.load(config_file)

    main(args)
