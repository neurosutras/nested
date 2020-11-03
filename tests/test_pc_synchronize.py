import sys, os
import time, datetime
import click

try:
    from mpi4py import MPI
    from neuron import h
except ImportError:
    raise ImportError('problem with importing neuron')
try:
    h.nrnmpi_init()
except Exception:
    print('h.nrnmpi_init() not executed; may not be defined in this version of NEURON')
    sys.stdout.flush()
    time.sleep(1.)


global_comm = MPI.COMM_WORLD
pc = h.ParallelContext()


def synchronize():
    print('Rank: %i reaching synchronize at %s' % (global_comm.rank, datetime.datetime.now()))
    if global_comm.rank == 0:
        test = {'test1': 1, 'test2': 2}
    else:
        test = None
    test = global_comm.bcast(test, root=0)
    print('Rank: %i; content of test dict: %s' % (global_comm.rank, str(test)))
    sys.stdout.flush()
    time.sleep(1.)


@click.command()
@click.option("--procs-per-worker", type=int, default=1)
def main(procs_per_worker):
    """

    :param procs_per_worker: int
    """

    pc.subworlds(procs_per_worker)
    global_rank = int(pc.id_world())
    global_size = int(pc.nhost_world())
    rank = int(pc.id())
    size = int(pc.nhost())
    print('MPI rank: %i, MPI size: %i, pc local rank: %i, pc local size: %i, pc global rank: %i, '
          'pc global size: %i\r' % (global_comm.rank, global_comm.size, rank, size, global_rank, global_size))
    sys.stdout.flush()
    time.sleep(1.)
    pc.runworker()
    pc.master_works_on_jobs(0)
    pc.context(synchronize)
    time.sleep(1.)
    print('Root has not yet entered synchronize at %s' % (datetime.datetime.now()))
    sys.stdout.flush()
    time.sleep(1.)
    synchronize()
    print('Root has exited synchronize at %s' % (datetime.datetime.now()))
    sys.stdout.flush()
    time.sleep(1.)
    pc.master_works_on_jobs(1)

    pc.done()
    h.quit()
    os._exit(1)


if __name__ == '__main__':
    main()
