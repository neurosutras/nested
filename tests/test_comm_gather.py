from mpi4py import MPI
import sys
import time
import click


@click.command()
@click.option("--procs-per-worker", type=int, default=1)
def main(procs_per_worker):
    """

    :param procs_per_worker: int
    """
    global_comm = MPI.COMM_WORLD

    global_rank = global_comm.rank
    global_size = global_comm.size

    num_worlds = int(global_size / procs_per_worker)

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
    print('global rank: %i, global size: %i, local rank: %i, local size: %i\r' %
          (global_comm.rank, global_comm.size, comm.rank, comm.size))
    sys.stdout.flush()
    time.sleep(1.)

    test = comm.gather(comm.rank, root=0)
    if comm.rank == 0:
        print('global rank: %i, global size: %i, local rank: %i, local size: %i\r' %
              (global_comm.rank, global_comm.size, comm.rank, comm.size))
        print(test)
    sys.stdout.flush()
    time.sleep(1.)


if __name__ == '__main__':
    main()
