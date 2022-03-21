import os
import subprocess
import numpy as np
import random
from abc import ABC, abstractclassmethod

from config import N1, N2, N3, NUM_ITER



def run(n1, n2, n3, simd, Olevel, num_thread, b1, b2, b3, num_iter=NUM_ITER):
    basename = 'iso3dfd_dev13_cpu'
    exec_name = basename + '_'+str(simd) + '_'+str(Olevel)+'.exe'
    p = subprocess.Popen([
        f"./iso3dfd-st7/bin/{exec_name}",
        str(n1),
        str(n2),
        str(n3),
        str(num_thread),
        str(num_iter),
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


def cost_fn(path):
    return -run(**dict(path[1:]))[1]


def neighborhood(levels, path, shuffle=True):
    """ Get neighbors solutions of a solution """
    idx_b1 = levels[-3][1].index(path[-3][1])
    idx_b2 = levels[-2][1].index(path[-2][1])
    idx_b3 = levels[-1][1].index(path[-1][1])
    neighbors = []
    for i1 in range(max(idx_b1-1, 0), min(idx_b1+2, len(levels[-3][1]))):
        for i2 in range(max(idx_b2-1, 0), min(idx_b2+2, len(levels[-2][1]))):
            for i3 in range(max(idx_b3-1, 0), min(idx_b3+2, len(levels[-1][1]))):
                neighbor_path = path.copy()
                neighbor_path[-3] = levels[-3][0], levels[-3][1][i1]
                neighbor_path[-2] = levels[-2][0], levels[-2][1][i2]
                neighbor_path[-1] = levels[-1][0], levels[-1][1][i3]
                neighbors.append(neighbor_path)
    if shuffle:
        random.shuffle(neighbors)
    return neighbors


class LocalSearch(ABC):
    def __call__(self, levels, pathes):
        better_pathes, costs =[], []
        for path in pathes:
            better_path, cost = self.search_fn(levels, path)
            better_pathes.append(better_path)
            costs.append(cost)
        return better_pathes, costs

    @abstractclassmethod
    def search_fn(self, levels, initial_solution):
        pass


class Identity(LocalSearch):
    def search_fn(self, levels, initial_solution):
        return initial_solution, cost_fn(initial_solution)


class GreedySearch(LocalSearch):
    def __init__(self, kmax=5):
        self.kmax = kmax

    def search_fn(self, levels, initial_solution):
        best_solution = initial_solution 
        best_cost = cost_fn(best_solution)
        neighbors = neighborhood(levels, best_solution, shuffle=False)
        k = 0
        new_best = True
        while (k < self.kmax and new_best):
            s = neighbors.pop()
            cost = cost_fn(s)
            for neigh in neighbors :
                new_cost = cost_fn(neigh)
                if new_cost < cost:
                    s = neigh
                    cost = new_cost
        
            if cost < best_cost :
                best_solution = s
                best_cost = cost
                neighbors = neighborhood(levels, best_solution, shuffle=False)
            else:
                new_best = False
            k = k+1
                
        return best_solution, best_cost


class SimulatedAnnealing(LocalSearch):
    def __init__(self, kmax=10, t0=1, update_t=lambda t: max(0.99*t, 0.1)):
        self.kmax = kmax
        self.t0 = t0
        self.update_t = update_t

    def search_fn(self, levels, initial_solution):
        t = self.t0
        best_solution = initial_solution
        best_cost = cost_fn(initial_solution)
        s = best_solution
        cost = best_cost
        neighbors = neighborhood(levels, s)
        k = 0
        while (k < self.kmax):
            s_prime = neighbors.pop()
            cost_prime = cost_fn(s_prime)
            if cost_prime < cost or random.random() < np.exp(-(cost_prime-cost)/t):
                s = s_prime
                cost = cost_prime
                neighbors = neighborhood(levels, s)
                if cost < best_cost:
                    print(cost, s)
                    best_solution = s
                    best_cost = cost
            t = self.update_t(t)
            k = k+1
        return best_solution, best_cost


class RandomizedTabu(LocalSearch):
    
    def __init__(self,tabu_size=5, kmax=10, k2max=10, t0=1, update_t=lambda t: max(0.99*t, 0.1)):
        self.kmax = kmax
        self.k2max = k2max
        self.t0 = t0
        self.update_t = update_t
        self.tabu_size = tabu_size

    def FifoAdd(self, solution, neigh_tabu):
        if len(neigh_tabu) == self.tabu_size:
            neigh_tabu.pop(0)
        neigh_tabu.append(solution)
        return neigh_tabu

    def search_fn(self, levels, initial_solution):
        t = self.t0
        best_solution = initial_solution
        best_cost = cost_fn(initial_solution)
        s = best_solution
        cost = best_cost
        neighbors = neighborhood(levels, s)
        neigh_tabu = [best_solution]
        k = 0
        k2 = 0

        while (k < self.kmax and k2 < self.k2max):
            s_prime = random.choice(neighbors)

            if s_prime not in neigh_tabu:
                cost_prime = cost_fn(s_prime)
                if cost_prime < cost or random.random() < np.exp(-(cost_prime-cost)/t):
                    s = s_prime
                    cost = cost_prime
                    neighbors = neighborhood(levels, s)
                    neigh_tabu = self.FifoAdd(s, neigh_tabu) 
                    if cost < best_cost:
                        print(cost, s)
                        best_solution = s
                        best_cost = cost
                t = self.update_t(t)
                k = k+1
                k2 = 0
            else:
                k2 = k2+1
        return best_solution, best_cost


if __name__ == "__main__":
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
    
    initial_path = [("init", "init"), ("simd", "avx"), ("Olevel", "-O2"), ("num_thread", 4), ("b1", 64), ("b2", 64), ("b3", 64)]
    # greedy_search(levels, initial_path, 5)
    print(len(neighborhood(levels, initial_path)))
    RandomizedTabu().search_fn(levels, initial_path)
