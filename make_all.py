import itertools
import os
import subprocess


def make(simd="avx2", Olevel="-O3"):
    os.chdir("./iso3dfd-st7/")
    os.system(f"make build simd={simd} Olevel={Olevel}")
    os.chdir("..")


def build(comb):
    for combination in comb:
        make(combination[0], combination[1])


if __name__ == '__main__':
    os.chdir("./iso3dfd-st7/")
    comb_build = list(itertools.product(["avx", "avx2", 'avx512', 'sse'],
                                        ["-O2", "-O3", "-Ofast"]))
    # to_build = []
    # for comb in comb_build:
    #     filename = 'iso3dfd_dev13_cpu_' + comb[0] + '_' + comb[1] + '.exe'
    #     if os.path.isfile(filename):
    #         pass
    #     else:
    #         to_build.append(comb)

    os.system(f"make clean")

    os.chdir("..")
    build(comb_build)
    # if len(to_build) > 0:
    #     build(to_build)
