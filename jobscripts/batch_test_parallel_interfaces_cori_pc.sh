#!/bin/bash -l

#SBATCH -J test_parallel_interfaces_pc_20180218
#SBATCH -o /global/homes/a/aaronmil/python_modules/nested/logs/test_parallel_interfaces_pc_20180218.%j.o
#SBATCH -e /global/homes/a/aaronmil/python_modules/nested/logs/test_parallel_interfaces_pc_20180218.%j.e
#SBATCH -q debug
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

# srun -N 32 -n 1024 -c 2 --cpu_bind=cores python test_parallel_interfaces.py --framework=pc
srun -N 3 -n 96 -c 2 --cpu_bind=cores python test_parallel_interfaces.py --framework=pc
