#! /bin/bash
#SBATCH --nodes=4
#SBATCH --job-name=test_mpi2
#SBATCH --partition=cpu_prod
#SBATCH --time=01:00:00
#SBATCH --qos=16nodespu
#SBATCH --exclusive
#SBATCH --ntasks 128

mpirun -np 4 -ppn 1 --bind-to socket python main.py --distibution exchange_all


