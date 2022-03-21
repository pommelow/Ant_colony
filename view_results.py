from operator import xor
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

label_l=["alpha","beta","rho","Q","nb_ant","method","identity","tau_min","tau_max","n_to_convert"]

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
            y_h[np.logical_and(t1 <= x, x < t2)] = c1 + a * (x[np.logical_and(t1 <= x, x < t2)] - t1)
        
        y += y_h
    
    y /= count
    return x, y

def mean_epoch(histories):
    max_time = min([h["time"][-1] for h in histories])
    min_time = max([h["time"][0] for h in histories])
    x = np.linspace(min_time, max_time, 1000)
    y = np.zeros_like(x)
    count = np.zeros_like(x)

    for h in histories:
  
        count[np.logical_and(h["time"][0] <= x, h["time"][-1] > x)] += 1
        y_h = np.zeros_like(x)

        for t1, t2, c1, c2 in zip(h["time"], h["time"][1:], h["epoch"], h["epoch"][1:]):
            a = (c2 - c1)
            y_h[np.logical_and(t1 <= x, x < t2)] = c1 + a 
        
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
        if histories!=[]:
            x, y = mean_curve(histories)
            plt.plot(x, y, label=config)
    
    plt.xlabel("Time")
    plt.ylabel("Best cost")
    plt.legend()
    plt.show()
    
def plot_parameter(folder,label):
    param=[]
    best_time=[]
    ref=[]
    for config in os.listdir(folder):
        labels=config.split("_")
        if (labels[label] not in ref) ^ (label==4 and labels[2]=="0.6"):   
            histories = get_histories(os.path.join(folder, config))
            ref.append(labels[label])
            if histories!=[]:
                x, y = mean_curve(histories)
                param.append(float(labels[label]))
                best_time.append(y[-10])



    best_time.sort(key=dict(zip(best_time, param)).get)
    param=sorted(param)
    plt.plot(param,best_time)
    plt.xlabel(label_l[label])
    plt.ylabel("Best cost")
    plt.xscale('log')
    plt.legend()
    plt.show()

def plot_epoch(folder):

    for config in os.listdir(folder):
        histories = get_histories(os.path.join(folder, config))
        if histories!=[]:
            x, y = mean_epoch(histories)
            plt.plot(x, y, label=config)
    plt.ylabel("epoch")
    #plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_parameter("./Results_mmas",label_l.index("tau_min"))
    #plot_epoch("./Results_conv")
    #plot_histories("./Results_mmas")
