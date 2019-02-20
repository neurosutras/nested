from nested.parallel import *
import click


context = Context()


def test(first, second, third=None):
    """

    :param first:
    :param second:
    :param third:
    :return:
    """
    if 'count' not in context():
        context.count = 0
    context.update(locals())
    context.count += 1
    time.sleep(0.2)
    return 'pid: %i, args: %s, count: %i' % (os.getpid(), str([first, second, third]), context.count)


def init_worker():
    """

    :return:
    """
    context.pid = os.getpid()
    return context.pid


@click.command()
@click.option("--cluster-id", type=str, default=None)
@click.option("--profile", type=str, default='default')
@click.option("--framework", type=click.Choice(['ipyp', 'pc', 'mpi']), default='ipyp')
@click.option("--procs-per-worker", type=int, default=1)
@click.option("--interactive", is_flag=True)
def main(cluster_id, profile, framework, procs_per_worker, interactive):
    """

    :param cluster_id: str
    :param profile: str
    :param framework: str
    :param procs_per_worker: int
    :param interactive: bool
    """
    if framework == 'ipyp':
        context.interface = IpypInterface(cluster_id=cluster_id, profile=profile,
                                          procs_per_worker=procs_per_worker, source_file=__file__)
        print 'before interface start: %i total processes detected' % context.interface.global_size
        try:
            result1 = context.interface.get('MPI.COMM_WORLD.size')
            print 'IpypInterface: ipengines each see an MPI.COMM_WORLD with size: %i' % max(result1)
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
    context.interface.start(disp=True)
    context.interface.ensure_controller()

    result2 = context.interface.apply(init_worker)
    sys.stdout.flush()
    time.sleep(1.)
    print 'after interface start: %i / %i workers participated in apply(init_worker)\n' % \
          (len(set(result2)), context.interface.num_workers)
    sys.stdout.flush()
    time.sleep(1.)

    time_stamp = time.time()
    start1 = 0
    end1 = start1 + int(context.interface.global_size)
    start2 = end1
    end2 = start2 + int(context.interface.global_size)
    print ': context.interface.map_sync(test, range(%i, %i), range(%i, %i))' % (start1, end1, start2, end2)
    pprint.pprint(context.interface.map_sync(test, range(start1, end1), range(start2, end2)))
    print '\n: map_sync took %.1f s\n' % (time.time() - time_stamp)
    sys.stdout.flush()
    time.sleep(1.)

    time_stamp = time.time()
    print ': context.interface.map_async(test, range(%i, %i), range(%i, %i))' % (start1, end1, start2, end2)
    result3 = context.interface.map_async(test, range(start1, end1), range(start2, end2))
    while not result3.ready(wait=0.1):
        pass
    result3 = result3.get()
    pprint.pprint(result3)
    sys.stdout.flush()
    print '\n: map_async took %.1f s\n' % (time.time() - time_stamp)
    sys.stdout.flush()
    time.sleep(1.)

    time_stamp = time.time()
    print ': context.interface.apply(test, 1, 2, third=3)'
    pprint.pprint(context.interface.apply(test, 1, 2, third=3))
    print '\n: apply took %.1f s\n' % (time.time() - time_stamp)
    sys.stdout.flush()
    time.sleep(1.)

    time_stamp = time.time()
    print ': context.interface.execute(init_worker)'
    result5 = context.interface.execute(init_worker)
    print '\n: execute returned: %s; took %.1f s\n' % (str(result5), time.time() - time_stamp)
    sys.stdout.flush()
    time.sleep(1.)

    time_stamp = time.time()
    print ': context.interface.get(\'context.pid\')'
    result4 = context.interface.get('context.pid')
    print '\n: get took %.1f s\n' % (time.time() - time_stamp)
    sys.stdout.flush()
    time.sleep(1.)
    print 'before interface stop: %i / %i workers participated in get operation\n' % \
          (len(set(result4)), context.interface.num_workers)
    sys.stdout.flush()
    time.sleep(1.)

    if not interactive:
        context.interface.stop()


if __name__ == '__main__':
    main(standalone_mode=False)
