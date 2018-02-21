#!/bin/bash -l

#SBATCH -J troubleshoot_ipyp_010418
#SBATCH -o troubleshoot_ipyp_010418.%j.o
#SBATCH -e troubleshoot_ipyp_010418.%j.e
#SBATCH -p debug
#SBATCH -N 3
#SBATCH -L SCRATCH
#SBATCH -C haswell
#SBATCH -t 0:30:00
#SBATCH --mail-user=aaronmil@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# source $HOME/.bash_profile
set -x

cd $HOME/python_modules/nested
export DATE=$(date +%Y%m%d_%H%M%S)
cluster_id="troubleshoot_ipyp_$DATE"

srun -N 1 -n 1 -c 2 --cpu_bind=cores ipcontroller --ip='*' --nodb --cluster-id=$cluster_id &
sleep 1
sleep 45
srun -N 1 -n 32 -c 2 --cpu_bind=cores ipengine --mpi=mpi4py --cluster-id=$cluster_id &
sleep 1
sleep 180
srun -N 1 -n 1 -c 2 --cpu_bind=cores python apply_example.py --cluster-id=$cluster_id
