#! /bin/bash
#SBATCH --nodes=2
#SBATCH --job-name=test_mpi2
#SBATCH --partition=cpu_inter
#SBATCH --time=01:00:00
#SBATCH --qos=16nodespu
#SBATCH --ntasks=64

mpirun -np 2 --map-by ppr:1:node --bind-to none python antcolony_mpi.py --config two_machines_test.json


