import itertools
import os


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
    os.system(f"make clean")

    os.chdir("..")
    build(comb_build)
