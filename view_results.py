import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def mean_curve(histories):
    max_time = min([h["time"][-1] for h in histories])
    min_time = max([h["time"][0] for h in histories])
    x = np.linspace(min_time, max_time, 1000)
    y = np.zeros_like(x)
    count = np.zeros_like(x)

    for h in histories:
        count[np.logical_and(h["time"][0] <= x, h["time"][-1] > x)] += 1
        y_h = np.zeros_like(x)

        for t1, t2, c1, c2 in zip(h["time"], h["time"][1:], h["best_cost"], h["best_cost"][1:]):
            a = (c2 - c1) / (t2 - t1)
            y_h[np.logical_and(t1 <= x, x <= t2)] = c1 + a * (x[np.logical_and(t1 <= x, x <= t2)] - t1)
        
        y += y_h
    
    y /= count
    return x, y


def get_histories(folder):
    histories = []
    for name in os.listdir(folder):
        filename = os.path.join(folder, name)
        with open(filename, "rb") as file:
            h = pickle.load(file)
            histories.append(h)

    return histories

def plot_histories(folder):
    for config in os.listdir(folder):
        histories = get_histories(os.path.join(folder, config))
        x, y = mean_curve(histories)
        if "identity_greedy" in config:
            y *= 10/16
        plt.plot(x, y, label=config)

    plt.xlabel("Time")
    plt.ylabel("Best cost")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_histories("./Results")
