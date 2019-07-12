#!/bin/bash -l

#SBATCH -J test_parallel_interfaces_pc_20180213
#SBATCH -o /global/homes/a/aaronmil/python_modules/nested/logs/test_parallel_interfaces_pc_20180213.%j.o
#SBATCH -e /global/homes/a/aaronmil/python_modules/nested/logs/test_parallel_interfaces_pc_20180213.%j.e
#SBATCH -p debug
#SBATCH -N 3
#SBATCH -L SCRATCH
#SBATCH -C knl,quad,cache
#SBATCH -S 4
#SBATCH -t 0:30:00
#SBATCH --mail-user=aaronmil@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# source $HOME/.bash_profile
set -x

cd $HOME/python_modules/nested
export DATE=$(date +%Y%m%d_%H%M%S)
cluster_id="troubleshoot_ipyp_$DATE"

srun -n 192 -c 4 --cpu-bind=cores python test_parallel_interfaces.py --framework=pc
