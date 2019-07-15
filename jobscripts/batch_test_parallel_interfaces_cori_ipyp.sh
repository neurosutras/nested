#!/bin/bash -l

#SBATCH -J test_parallel_interfaces_ipyp_20180216
#SBATCH -o /global/homes/a/aaronmil/python_modules/nested/logs/test_parallel_interfaces_ipyp_20180216.%j.o
#SBATCH -e /global/homes/a/aaronmil/python_modules/nested/logs/test_parallel_interfaces_ipyp_20180216.%j.e
#SBATCH -q debug
#SBATCH -N 64
#SBATCH -L SCRATCH
#SBATCH -C haswell
#SBATCH -t 0:30:00
#SBATCH --mail-user=aaronmil@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# source $HOME/.bash_profile
set -x

cd $HOME/python_modules/nested
export DATE=$(date +%Y%m%d_%H%M%S)
cluster_id="test_parallel_interfaces_ipyp_$DATE"

ulimit -n 20000
ulimit -u 20000

srun -N 1 -n 1 -c 2 --cpu-bind=cores ipcontroller --ip='*' --nodb --HeartMonitor.max_heartmonitor_misses=50 --HubFactory.registration_timeout=150 --log-level=DEBUG --cluster-id=$cluster_id &
sleep 1
sleep 60
srun -N 62 -n 1984 -c 2 --cpu-bind=cores ipengine --mpi=mpi4py --EngineFactory.timeout=30 --cluster-id=$cluster_id &
sleep 1
sleep 360
srun -N 1 -n 1 -c 2 --cpu-bind=cores python test_parallel_interfaces.py --cluster-id=$cluster_id --framework=ipyp
