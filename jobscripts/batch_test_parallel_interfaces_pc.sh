#!/bin/bash -l

#SBATCH -J test_parallel_interfaces_pc_20180131
#SBATCH -o /global/homes/a/aaronmil/python_modules/nested/logs/test_parallel_interfaces_pc_20180131.%j.o
#SBATCH -e /global/homes/a/aaronmil/python_modules/nested/logs/test_parallel_interfaces_pc_20180131.%j.e
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

srun -N 3 -n 96 -c 2 python test_parallel_interfaces.py --framework=pc
