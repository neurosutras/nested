#!/bin/bash -l

#SBATCH -J test_parallel_interfaces_KNL_pc_20180914
#SBATCH -o /global/homes/a/aaronmil/python_modules/nested/logs/test_parallel_interfaces_KNL_pc_20180914.%j.o
#SBATCH -e /global/homes/a/aaronmil/python_modules/nested/logs/test_parallel_interfaces_KNL_pc_20180914.%j.e
#SBATCH -q debug
#SBATCH -N 2
#SBATCH -L SCRATCH
#SBATCH -C knl,quad,cache
#SBATCH -S 4
#SBATCH -t 0:10:00
#SBATCH --mail-user=aaronmil@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $HOME/python_modules/nested

srun -n 64 -c 8 --cpu-bind=cores python test_parallel_interfaces.py --framework=pc --procs-per-worker=2
