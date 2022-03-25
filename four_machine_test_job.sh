#! /bin/bash
#SBATCH --nodes=4
#SBATCH --job-name=test_mpi2
#SBATCH --partition=cpu_prod
#SBATCH --time=01:00:00
#SBATCH --qos=16nodespu
#SBATCH --exclusive
#SBATCH --ntasks 128

mpirun -np 4 --map-by ppr:1:node --bind-to socket  python /usr/users/cpust75/cpust75_6/Ant_colony/antcolony_mpi.py --config four_machine_test.json


