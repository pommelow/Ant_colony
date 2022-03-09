import itertools
import os
import subprocess


def make(simd="avx2", Olevel="-O3"):
    os.chdir("./iso3dfd-st7/")
    os.system(f"make build simd={simd} Olevel={Olevel}")
    os.chdir("..")

    # os.chdir("./iso3dfd-st7/")
    # # try:
    # #     pass
    # #     # os.system(f"make clean")
    # #     # subprocess.run(["make", "clean"],
    # #     #                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # # except Exception as e:
    # #     print(e)
    # #     pass
    # # subprocess.run(["make", "build", f"simd={simd}", f" Olevel={Olevel} "],
    # #                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # os.system(f"make build simd={simd} Olevel={Olevel}")

    # os.chdir("..")


def build(comb):
    for combination in comb:
        make(combination[0], combination[1])

if __name__ == '__main__':
    os.chdir("./iso3dfd-st7/")
    os.system(f"make clean")
    os.chdir("..")

    comb_build = list(itertools.product(["avx", "avx2", 'avx512', 'sse'],
                                    ["-O2", "-O3", "-Ofast"]))
    build(comb_build)
