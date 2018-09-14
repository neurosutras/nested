import sys
import pprint
import time
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print 'Rank: %i sees size: %i before executor is built. (%s)' % (rank, size, __name__)


def do_work(i):
    if rank == 0:
        # Master process executes code below
        open_ranks = range(1, size)
        for worker_rank in open_ranks:
            future = comm.irecv(source=worker_rank)
            val = future.wait()
            # print rank, worker_rank, val
        for worker_rank in open_ranks:
            comm.isend(0, dest=worker_rank)
    else:
        # Worker processes execute code below
        # print("I am a worker with rank {}.".format(rank))
        comm.isend(rank, dest=0)
        future = comm.irecv(source=0)
        val = future.wait()
        return rank


def main():
    num_workers = size - 1
    num_tasks = 0
    for i in xrange(3):
        start_time = time.time()
        tasks = range(num_tasks, num_tasks + num_workers)
        executor = MPIPoolExecutor()
        future_list = executor.map(do_work, tasks)
        returned_ranks = []
        do_work(None)
        for result in future_list:
            returned_ranks.append(result)
        used_workers = len(set(returned_ranks))
        print 'Map: %i used %i/%i unique workers and took %.4f s' % \
              (i, used_workers, num_workers, time.time() - start_time)
        if used_workers != num_workers:
            pprint.pprint(returned_ranks)


if __name__ == "__main__":
    main()