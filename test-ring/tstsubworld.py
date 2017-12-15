from mpi4py import MPI
from neuron import h
import click


@click.command()
@click.option("--subworlds", type=int, default=1)
def main(subworlds):
    """

    :param subworlds: int
    :return:
    """
    pc = h.ParallelContext()
    pc.subworlds(subworlds)

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    print 'rank: ({6}, {7}, world: ({0}, {1}) bbs ({2}, {3}) net ({4}, {5})'.format(
        pc.id_world(), pc.nhost_world(), pc.id_bbs(), pc.nhost_bbs(), pc.id(), pc.nhost()
    )

    (s)

    from ring import runring

    pc.runworker()

    for ncell in range(5, 10):
      pc.submit(runring, ncell, 1, 100)

    while (pc.working()):
      print(pc.pyret())

    pc.done()
    h.quit()


if __name__ == '__main__':
    main()
