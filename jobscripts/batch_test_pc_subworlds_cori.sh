#!/bin/bash -l

#SBATCH -J test_pc_subworlds_20180220
#SBATCH -o /global/homes/a/aaronmil/python_modules/nested/logs/test_pc_subworlds_20180220.%j.o
#SBATCH -e /global/homes/a/aaronmil/python_modules/nested/logs/test_pc_subworlds_20180220.%j.e
#SBATCH -p debug
#SBATCH -N 3  # 32
#SBATCH -L SCRATCH
#SBATCH -C haswell
#SBATCH -t 0:30:00
#SBATCH --mail-user=aaronmil@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $HOME/python_modules/nested
export DATE=$(date +%Y%m%d_%H%M%S)

ulimit -n 20000
ulimit -u 20000

# srun -N 32 -n 1024 -c 2 --cpu_bind=cores python test_pc_subworlds.py --test-subworlds --procs-per-worker=2
srun -N 3 -n 96 -c 2 --cpu_bind=cores python test_pc_subworlds.py --test-subworlds --procs-per-worker=2
