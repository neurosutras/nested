from __future__ import print_function
from past.builtins import xrange
import sys
import pprint
import time
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print('Rank: %i sees size: %i before executor is built. (%s)' % (rank, size, __name__))


def do_work(i):
    if rank == 1:
        open_ranks = range(2, size)
        for worker_rank in open_ranks:
            future = comm.irecv(source=worker_rank)
            val = future.wait()
        for worker_rank in open_ranks:
            comm.isend(1, dest=worker_rank)
    elif rank > 1:
        comm.isend(rank, dest=1)
        future = comm.irecv(source=1)
        val = future.wait()
    group = comm.Get_group()
    sub_group = group.Incl(range(1, comm.size))
    worker_comm = comm.Create(sub_group)
    return rank


def main():
    num_workers = size - 1
    print('Rank: %i sees main' % (rank))
    for i in xrange(3):
        with MPICommExecutor(comm, root=0) as executor:
            start_time = time.time()
            tasks = range(num_workers)
            future_list = executor.map(do_work, tasks)
            do_work(0)
        returned_ranks = []
        for result in future_list:
            returned_ranks.append(result)
        used_workers = len(set(returned_ranks))
        print('Map: %i used %i/%i unique workers and took %.4f s' % \
              (i, used_workers, num_workers, time.time() - start_time))
        if used_workers != num_workers:
            pprint.pprint(returned_ranks)


if __name__ == "__main__":
    main()