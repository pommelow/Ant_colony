import itertools
import os

comb_build = list(itertools.product(["avx", "avx2", 'avx512', 'sse'],
                                    ["-O2", "-O3", "-Ofast"]))


def make(simd="avx2", Olevel="-O3"):
    os.chdir("./iso3dfd-st7/")
    try:
        os.system(f"make clean")
    except Exception as e:
        print(e)
        pass
    os.system(f"make build simd={simd} Olevel={Olevel}")
    os.chdir("..")


def build_change(comb):
    for combination in comb:
        make(comb[0], comb[1])
