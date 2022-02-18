import os
import subprocess


def make(simd="avx2", Olevel="O3"):
    os.system(f"make build simd={simd} Olevel={Olevel}")

def run(n1, n2, n3, num_thread, iteration, b1, b2, b3):
    filename = os.listdir("~/Documents/iso3dfd-st7/bin/")
    p = subprocess.Popen([
        f"~/Documents/iso3dfd-st7/bin/{filename}",
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
    time = float(outputs[0].split(" ")[-2])
    throughput = float(outputs[1].split(" ")[-2])
    flops = float(outputs[2].split(" ")[-2])
    return time, throughput, flops


class AntColony():

    class Ant():
        def __init__(self) -> None:
            self.history = None
            self.path = []
            self.state = "initial state"

        def pick_path(self):
            return next_state

    def __init__(self, alpha, beta, rho, Q):
        levels = {"simd": {"avx", "avx2"}, "Olevel": {"O2", "O3"}}
        tau = dict()
        pass

    def make(self, config):
        """ Compile the project """
        pass

    def run(self, config):
        """ Run the executable and return the time"""
        subprocess.run("file" + config)
        pass




