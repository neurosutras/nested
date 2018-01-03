from parallel import *
import click


context = Context()


def collect_ranks(tag):
    """

    :param tag: int
    :return: tuple: int, dict
    """
    time.sleep(0.1)
    start_time = time.time()
    ranks = context.interface.comm.gather(context.interface.global_rank, root=0)
    if 'count' not in context():
        context.count = 0
    context.count += 1
    if context.interface.rank == 0:
        return 'worker_id: %i, global_ranks: %s, tag: %i, count: %i, compute time: %.2f (ms)' % \
               (context.interface.worker_id, str(ranks), int(tag), context.count, (time.time() - start_time) * 1000.)


def set_count(count=None):
    """

    :param count: int
    """
    if count is None:
        if 'count' not in context():
            context.count = 0
        context.count += 1
    else:
        context.count = count
    print 'global rank: %i / %i, local rank: %i / %i within subworld %i / %i, count: %i' % \
          (context.interface.global_rank, context.interface.global_size, context.interface.rank, context.interface.size,
           context.interface.worker_id, context.interface.num_workers, context.count)


@click.command()
@click.option("--procs-per-worker", type=int, default=1)
@click.option("--test-subworlds", isflag=True)
def main(procs_per_worker, test_subworlds):
    """

    :param procs_per_worker: int
    :param test_subworlds
    """
    
    try:
        from mpi4py import MPI
        from neuron import h
    except ImportError:
        raise ImportError('nested: ParallelContextInterface: problem with importing neuron')
    global_comm = MPI.COMM_WORLD
    pc = h.ParallelContext()
    if test_subworlds:
        pc.subworlds(procs_per_worker)
    global_rank = int(pc.id_world())
    global_size = int(pc.nhost_world())
    rank = int(pc.id())
    size = int(pc.nhost())
    for i in xrange(global_comm.size):
        if global_comm.rank == i:
            print 'MPI rank: %i, MPI size: %i, pc local rank: %i, pc local size: %i, pc global rank: %i, ' \
                  'pc global size: %i' % (global_comm.rank, global_comm.size, rank, size, global_rank, global_size)
            time.sleep(1.)
        else:
            time.sleep(1.)


if __name__ == '__main__':
    main()
