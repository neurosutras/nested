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
def main(procs_per_worker):
    """

    :param procs_per_worker: int
    """
    context.interface = ParallelContextInterface(procs_per_worker=procs_per_worker)
    if context.interface.global_rank == 0:
        print 'before interface.start()\n: context.interface.apply(set_count)'
    results1 = context.interface.apply(set_count)
    time.sleep(0.1)
    if context.interface.global_rank == 0:
        pprint.pprint(results1)
    time.sleep(0.1)
    context.interface.start()
    print 'after interface.start()\n: context.interface.apply(set_count, 5)'
    results2 = context.interface.apply(set_count, 5)
    time.sleep(0.1)
    pprint.pprint(results2)
    time.sleep(0.1)
    print ': context.interface.map_sync(collect_ranks, range(10))'
    results3 = context.interface.map_sync(collect_ranks, range(10))
    time.sleep(0.1)
    pprint.pprint(results3)
    time.sleep(0.1)
    print ': context.interface.map_async(collect_ranks, range(10, 20))'
    results4 = context.interface.map_async(collect_ranks, range(10, 20))
    print 'collected result keys: %s' % str(context.interface.collected.keys())
    while not results4.ready():
        pass
    time.sleep(0.1)
    results4 = results4.get()
    pprint.pprint(results4)
    time.sleep(0.1)
    print 'collected result keys: %s' % str(context.interface.collected.keys())
    time.sleep(0.1)
    print ': context.interface.apply(collect_ranks, 0)'
    results5 = context.interface.apply(collect_ranks, 0)
    time.sleep(0.1)
    pprint.pprint(results5)
    time.sleep(0.1)
    context.interface.stop()


if __name__ == '__main__':
    main()