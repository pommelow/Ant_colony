#! /bin/bash
#SBATCH --nodes=4
#SBATCH --job-name=test_mpi2
#SBATCH --partition=cpu_inter
#SBATCH --time=01:00:00
#SBATCH --qos=16nodespu
#SBATCH --ntasks=128

mpirun -np 4 --map-by ppr:1:node --bind-to socket python antcolony_mpi.py -m 0


