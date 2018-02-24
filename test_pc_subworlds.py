import sys
import time
import click


@click.command()
@click.option("--procs-per-worker", type=int, default=1)
def main(procs_per_worker):
    """

    :param procs_per_worker: int
    """
    
    try:
        from mpi4py import MPI
        from neuron import h
    except ImportError:
        raise ImportError('nested: ParallelContextInterface: problem with importing neuron')
    global_comm = MPI.COMM_WORLD
    pc = h.ParallelContext()
    pc.subworlds(procs_per_worker)
    global_rank = int(pc.id_world())
    global_size = int(pc.nhost_world())
    rank = int(pc.id())
    size = int(pc.nhost())
    print 'MPI rank: %i, MPI size: %i, pc local rank: %i, pc local size: %i, pc global rank: %i, ' \
          'pc global size: %i\r' % (global_comm.rank, global_comm.size, rank, size, global_rank, global_size)
    sys.stdout.flush()
    time.sleep(1.)


if __name__ == '__main__':
    main()
