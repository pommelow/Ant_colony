#! /bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks=128
#SBATCH --job-name=test_mpi_parallel
#SBATCH --partition=cpu_prod
#SBATCH --time=01:00:00
#SBATCH --qos=16nodespu

mpirun -np 4 -map-by ppr:1:node -bind-to socket python3 antcolony_mpi.py -m 0


