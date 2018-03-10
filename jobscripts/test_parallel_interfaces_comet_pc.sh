#!/bin/sh
#
#SBATCH -J test_parallel_interfaces_comet_pc_20180224
#SBATCH -o /home/aaronmil/python_modules/nested/logs/test_parallel_interfaces_comet_pc_20180224.%j.o
#SBATCH -e /home/aaronmil/python_modules/nested/logs/test_parallel_interfaces_comet_pc_20180224.%j.e
#SBATCH -N 2 -n 48
#SBATCH --export=ALL
#SBATCH --tasks-per-node=24
#SBATCH -p debug
#SBATCH -t 00:10:00

set -x

cd $HOME/python_modules/nested

export LD_PRELOAD=$MPIHOME/lib/libmpi.so

ibrun -n 48 python test_parallel_interfaces.py --framework=pc --procs-per-worker=2
