#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=test_mpi_parallel
#SBATCH --partition=cpu_inter
#SBATCH --time=00:30:00
#SBATCH --qos=1nodespu

#mpirun -np 1 python antcolony_mpi.py -m 0

srun -N 1 --reservation SHPI1 --pty bash #antcolony_mpi.py -m 0 


