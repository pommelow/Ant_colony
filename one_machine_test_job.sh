#! /bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=test_mpi2
#SBATCH --partition=cpu_inter
#SBATCH --time=01:00:00
#SBATCH --qos=16nodespu
#SBATCH --ntasks=16

mpirun -np 1 --map-by ppr:1:node --bind-to none python antcolony_mpi.py --config one_machine_test.json


