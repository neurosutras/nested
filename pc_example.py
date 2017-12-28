from utils import *
from parallel import *
import click


script_filename = 'pc_example.py'

context = Context()


def collect_ranks(tag):
    """

    :param tag: int
    :return: tuple: int, dict
    """
    time.sleep(0.1)
    start_time = time.time()
    ranks = context.interface.comm.gather(context.interface.global_rank, root=0)
    if context.interface.rank == 0:
        return context.interface.worker_id, {'ranks': ranks, 'tag': int(tag), 'compute time': time.time() - start_time}


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
    print ': context.interface.apply(set_count)'
    context.interface.apply(set_count)
    time.sleep(0.1)
    print ': context.interface.apply(set_count, 5)'
    context.interface.apply(set_count, 5)
    time.sleep(0.1)
    context.interface.start()
    results1 = context.interface.map_sync(collect_ranks, range(10))
    print ': pprint.pprint(results1)'
    pprint.pprint(results1)

    results2 = context.interface.map_async(collect_ranks, range(10, 20))
    results3 = context.interface.map_async(collect_ranks, range(20, 30))
    start_time = time.time()

    print 'collected result keys: %s' % str(context.interface.collected.keys())
    while not results2.ready():
        pass
    results2 = results2.get()
    print ': pprint.pprint(results2)'
    pprint.pprint(results2)
    print 'collected result keys: %s' % str(context.interface.collected.keys())
    results4 = context.interface.collect_results()
    print ': pprint.pprint(results4)'
    pprint.pprint(results4)
    print 'collected result keys: %s' % str(context.interface.collected.keys())
    while not results3.ready():
        pass
    try:
        results3 = results3.get()
        print ': pprint.pprint(results3)'
        pprint.pprint(results3)
    except KeyError:
        print 'results3.get() failed, because results4 = context.interface.collect_results() cleared the internal ' \
              'buffer'
        print 'result3 keys: %s' % results3.keys
    print 'collected result keys: %s' % str(context.interface.collected.keys())
    context.interface.stop()
    # h.quit()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_filename) != -1, sys.argv) + 1):])