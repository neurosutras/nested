from nested.parallel import *
import click


context = Context()


def test(arg):
    """

    :param arg: int
    :return:
    """
    if 'count' not in context():
        context.count = 0
    context.update(locals())
    context.count += 1
    # time.sleep(0.05)
    return context.interface.global_rank, arg, context.count


def init_worker():
    """

    :return:
    """
    context.pid = os.getpid()
    if 'interface' in context():
        context.interface.start(disp=True)
        context.interface.ensure_controller()
    return context.pid


@click.command()
@click.option("--cluster-id", type=str, default=None)
@click.option("--profile", type=str, default='default')
@click.option("--framework", type=click.Choice(['ipyp', 'pc', 'mpi']), default='pc')
@click.option("--procs-per-worker", type=int, default=1)
@click.option("--map_len", type=int, default=None)
def main(cluster_id, profile, framework, procs_per_worker, map_len):
    """

    :param cluster_id: str
    :param profile: str
    :param framework: str
    :param procs_per_worker: int
    :param map_len: int
    """
    if framework == 'ipyp':
        context.interface = IpypInterface(cluster_id=cluster_id, profile=profile,
                                          procs_per_worker=procs_per_worker, source_file=__file__)
        print 'before interface start: %i total processes detected' % context.interface.global_size
        try:
            result0 = context.interface.get('MPI.COMM_WORLD.size')
            print 'IpypInterface: ipengines each see an MPI.COMM_WORLD with size: %i' % max(result0)
        except Exception:
            print 'IpypInterface: ipengines do not see an MPI.COMM_WORLD'
    elif framework == 'pc':
        context.interface = ParallelContextInterface(procs_per_worker=procs_per_worker)
        result1 = context.interface.get('context.interface.global_rank')
        if context.interface.global_rank == 0:
            print 'before interface start: %i / %i total processes participated in get operation' % \
                  (len(set(result1)), context.interface.global_size)
    elif framework == 'mpi':
        context.interface = MPIFuturesInterface(procs_per_worker=procs_per_worker)
        result1 = context.interface.get('context.global_comm.rank')
        print 'before interface start: %i / %i workers participated in get operation' % \
                  (len(set(result1)), context.interface.num_workers)
    sys.stdout.flush()
    time.sleep(1.)

    result2 = context.interface.apply(init_worker)
    sys.stdout.flush()
    time.sleep(1.)
    print 'after interface start: %i / %i workers participated in apply(init_worker)\n' % \
          (len(set(result2)), context.interface.num_workers)
    sys.stdout.flush()
    time.sleep(1.)

    time_stamp = time.time()
    if map_len is None:
        # map_len = int(context.interface.global_size * 1000)
        map_len = 2147483647  # c++ maxint
    map_inc = 10000
    start = 0
    end = map_inc
    while end < map_len:
        result3 = context.interface.map_async(test, range(start, end))
        print np.array(result3.keys[-10:])
        while not result3.ready():
            pass
        result3 = result3.get()
        start += map_inc
        end += map_inc
    print ': context.interface.map_async(test, range(%i, %i)' % (start, end)
    result3 = context.interface.map_async(test, range(start, end))
    print ': result3.keys()'
    print np.array(result3.keys[-10:])
    while not result3.ready():
        pass
    result3 = result3.get()[-10:]
    pprint.pprint(result3)
    sys.stdout.flush()
    print '\n: map_async took %.1f s\n' % (time.time() - time_stamp)
    sys.stdout.flush()
    time.sleep(1.)

    context.interface.stop()


if __name__ == '__main__':
    main(standalone_mode=False)
