#! /bin/bash
#SBATCH --nodes=2
#SBATCH --job-name=test_mpi2
#SBATCH --partition=cpu_tp
#SBATCH --time=02:00:00

mpirun -np 2 python antcolony_mpi.py -m 0


