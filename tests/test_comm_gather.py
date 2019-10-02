import os
import sys
import time
import click
import pprint
from nested.utils import Context

context = Context()


def test_gather(*args, **kwargs):
    # test = context.comm.gather(context.global_rank, root=0)
    test = context.global_rank
    time.sleep(1.)
    context.count += 1
    if context.rank == 0:
        return test


@click.command()
@click.option("--procs-per-worker", type=int, default=1)
def main(procs_per_worker):
    """

    :param procs_per_worker: int
    """
    try:
        from mpi4py import MPI
    except ImportError:
        raise ImportError('nested: problem with importing from mpi4py')

    global_comm = MPI.COMM_WORLD

    global_rank = global_comm.rank
    global_size = global_comm.size

    size = procs_per_worker
    num_worlds = int(global_size / procs_per_worker)
    rank = global_rank % size

    count = 0

    all_global_ranks = list(range(global_size))
    global_ranks = None
    for i in range(num_worlds):
        start = i * procs_per_worker
        these_global_ranks = all_global_ranks[start:start+procs_per_worker]
        if global_rank in these_global_ranks:
            global_ranks = these_global_ranks

    group = global_comm.Get_group()
    sub_group = group.Incl(global_ranks)
    comm = global_comm.Create(sub_group)
    print('MPI rank: %i, MPI size: %i, pc local rank: %i, pc local size: %i, pc global rank: %i, '
          'pc global size: %i\r' % (global_comm.rank, global_comm.size, rank, size, global_rank, global_size))
    sys.stdout.flush()
    time.sleep(1.)
    context.update(locals())

    test = comm.gather(comm.rank, root=0)
    if comm.rank == 0:
        print('MPI rank: %i, MPI size: %i, pc local rank: %i, pc local size: %i, pc global rank: %i, '
              'pc global size: %i\r' % (global_comm.rank, global_comm.size, rank, size, global_rank, global_size))
        print(test)
    sys.stdout.flush()
    time.sleep(1.)

if __name__ == '__main__':
    main()
