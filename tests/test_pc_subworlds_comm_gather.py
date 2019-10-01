import os
import sys
import time
import click
import pprint
from nested.utils import Context

context = Context()


def test_gather(*args, **kwargs):
    test = context.comm.gather(context.global_rank, root=0)
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
        from neuron import h
    except ImportError:
        raise ImportError('nested: ParallelContextInterface: problem with importing neuron')
    try:
        h.nrnmpi_init()
    except Exception:
        print('nested: ParallelContextInterface: h.nrnmpi_init() not executed; may not be defined in this version '
              'of NEURON')
        sys.stdout.flush()
        time.sleep(1.)
    global_comm = MPI.COMM_WORLD
    pc = h.ParallelContext()
    pc.subworlds(procs_per_worker)
    count = 0
    global_rank = int(pc.id_world())
    global_size = int(pc.nhost_world())
    rank = int(pc.id())
    size = int(pc.nhost())
    global_ranks = [global_rank] * size
    global_ranks = pc.py_alltoall(global_ranks)
    group = global_comm.Get_group()
    sub_group = group.Incl(global_ranks)
    comm = global_comm.Create(sub_group)
    worker_id = comm.bcast(int(pc.id_bbs()), root=0)
    num_workers = comm.bcast(int(pc.nhost_bbs()), root=0)
    print('MPI rank: %i, MPI size: %i, pc local rank: %i, pc local size: %i, pc global rank: %i, '
          'pc global size: %i\r' % (global_comm.rank, global_comm.size, rank, size, global_rank, global_size))
    sys.stdout.flush()
    time.sleep(1.)
    context.update(locals())
    pc.runworker()
    print('After pc.runworker:')
    print('MPI rank: %i, MPI size: %i, pc local rank: %i, pc local size: %i, pc global rank: %i, '
          'pc global size: %i\r' % (global_comm.rank, global_comm.size, rank, size, global_rank, global_size))
    sys.stdout.flush()
    time.sleep(1.)

    keys = list(range(10))
    for key in keys:
        pc.submit(key, test_gather, key)
    vals = {}
    for key in keys:
        while pc.working():
            key = int(pc.userid())
            val = pc.pyret()
            vals[key] = val
    if global_rank == 0:
        print('test_gather results:')
        pprint.pprint(list(vals.items()))

    pc.done()
    h.quit()
    os._exit(1)


if __name__ == '__main__':
    main()
