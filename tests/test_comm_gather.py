from mpi4py import MPI
import sys
import time

comm = MPI.COMM_WORLD

rank = comm.rank
size = comm.size

test = None
test = comm.gather(rank, root=0)
if comm.rank == 0:
    print('root: rank: %i, size: %i\r' % (rank, size))
    print(test)
sys.stdout.flush()
time.sleep(1.)
