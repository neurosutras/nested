from nested.parallel import *
import click


context_monkeys = Context()


def test(first, second, third=None):
    # 20180219: debugging recursion depth error on Cori
    if context_monkeys.interface_monkeys.global_rank == 0:
        print context_monkeys()
    if 'count' not in context_monkeys():
        context_monkeys.count = 0
    context_monkeys.update(locals())
    context_monkeys.count += 1
    time.sleep(0.2)
    return 'pid: %i, args: %s, count: %i' % (os.getpid(), str([first, second, third]), context_monkeys.count)


def init_worker():
    try:
        context_monkeys.interface_monkeys.start(disp=True)
    except Exception:
        pass
    return 'test'


@click.command()
@click.option("--cluster-id", type=str, default=None)
@click.option("--profile", type=str, default='default')
@click.option("--framework", type=click.Choice(['ipyp', 'pc']), default='ipyp')
@click.option("--procs-per-worker", type=int, default=1)
def main(cluster_id, profile, framework, procs_per_worker):
    """

    :param cluster_id: str
    :param profile: str
    :param framework: str
    :param procs_per_worker: int
    """
    if framework == 'ipyp':
        context_monkeys.interface_monkeys = IpypInterface(cluster_id=cluster_id, profile=profile,
                                                          procs_per_worker=procs_per_worker, source_file=__file__)
    elif framework == 'pc':
        context_monkeys.interface_monkeys = ParallelContextInterface(procs_per_worker=procs_per_worker)
        result1 = context_monkeys.interface_monkeys.get('context_monkeys.interface_monkeys.global_rank')
        if context_monkeys.interface_monkeys.global_rank == 0:
            print result1
        time.sleep(0.1)
    print ': context_monkeys.interface_monkeys.apply(init_worker)'
    print context_monkeys.interface_monkeys.apply(init_worker)
    context_monkeys.interface_monkeys.ensure_controller()
    # print ': context_monkeys.interface_monkeys.apply(test, 1, 2, third=3)'
    # print context_monkeys.interface_monkeys.apply(test, 1, 2, third=3)
    # print ': context_monkeys.interface_monkeys.get(\'context_monkeys.count\')'
    # print context_monkeys.interface_monkeys.get('context_monkeys.count')    
    start1 = 0
    end1 = start1 + int(context_monkeys.interface_monkeys.global_size)
    start2 = end1
    end2 = start2 + int(context_monkeys.interface_monkeys.global_size)
    print ': context_monkeys.interface_monkeys.map_sync(test, range(%i, %i), range(%i, %i))' % (start1, end1, start2, end2)
    print context_monkeys.interface_monkeys.map_sync(test, range(start1, end1), range(start2, end2))
    print ': context_monkeys.interface_monkeys.map_async(test, range(%i, %i), range(%i, %i))' % (start1,end1, start2, end2)
    result2 =  context_monkeys.interface_monkeys.map_async(test, range(start1, end1),range(start2, end2))
    while not result2.ready():
        time.sleep(0.1)
    result2 = result2.get()
    pprint.pprint(result2)
    print ': context_monkeys.interface_monkeys.get(\'context_monkeys.count\')'
    print context_monkeys.interface_monkeys.get('context_monkeys.count')
    context_monkeys.interface_monkeys.stop()


if __name__ == '__main__':
    main(standalone_mode=False)
