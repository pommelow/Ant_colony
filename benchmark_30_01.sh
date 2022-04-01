#! /bin/bash
#SBATCH --nodes=8
#SBATCH --job-name=test_mpi2
#SBATCH --partition=cpu_prod
#SBATCH --time=02:00:00
#SBATCH --qos=16nodespu
#SBATCH --exclusive
#SBATCH --ntasks 256

mpirun -np 8 -ppn 1 --bind-to socket python main.py --distribution exchange_all --local-search identity --nb_ant 15 --tau_min 0.5 --tau_max 10 --nb-to-update 15


