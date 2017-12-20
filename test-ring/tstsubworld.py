from mpi4py import MPI
from neuron import h
import click
import time


@click.command()
@click.option("--subworld-size", type=int, default=1)
def main(subworld_size):
    """

    :param subworlds: int
    :return:
    """
    pc = h.ParallelContext()
    pc.subworlds(subworld_size)

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    print 'rank: ({6}, {7}), world: ({0}, {1}), bbs: ({2}, {3}), net: ({4}, {5})'.format(
        pc.id_world(), pc.nhost_world(), pc.id_bbs(), pc.nhost_bbs(), pc.id(), pc.nhost(), rank, size)

    from ring import runring

    pc.runworker()

    for i in xrange(5):
        pc.submit(runring, 6, 1, 100)
        time.sleep(0.1)

    while (pc.working()):
        print(pc.pyret())

    pc.done()
    # h.quit()


if __name__ == '__main__':
    main()
