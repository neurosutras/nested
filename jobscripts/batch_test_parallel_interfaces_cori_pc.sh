#!/bin/bash -l

#SBATCH -J test_parallel_interfaces_pc_20180224
#SBATCH -o /global/homes/a/aaronmil/python_modules/nested/logs/test_parallel_interfaces_pc_20180224.%j.o
#SBATCH -e /global/homes/a/aaronmil/python_modules/nested/logs/test_parallel_interfaces_pc_20180224.%j.e
#SBATCH -q debug
#SBATCH -N 6
#SBATCH -L SCRATCH
#SBATCH -C haswell
#SBATCH -t 0:30:00
#SBATCH --mail-user=aaronmil@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $HOME/python_modules/nested
export DATE=$(date +%Y%m%d_%H%M%S)

srun -N 6 -n 192 -c 2 python test_parallel_interfaces.py --framework=pc
