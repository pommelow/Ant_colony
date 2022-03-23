#! /bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --job-name=test_mpi_parallel
#SBATCH --partition=cpu_prod
#SBATCH --time=01:00:00
#SBATCH --qos=16nodespu

mpirun -np 2 -map-by ppr:1:node -bind-to socket python antcolony_mpi.py -m 0


