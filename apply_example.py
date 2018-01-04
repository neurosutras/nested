from nested.parallel import *
import click


context_monkeys = Context()


def test(first, second, third=None):
    pid = os.getpid()
    print pid, first, second, third
    time.sleep(0.3)
    context_monkeys.first = first
    return pid, first, second, third


def init_worker():
    try:
        context_monkeys.interface_monkeys.start(disp=True)
    except Exception:
        pass
    # return 'test'


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
        context_monkeys.interface_monkeys.start(disp=True)
    elif framework == 'pc':
        context_monkeys.interface_monkeys = ParallelContextInterface(procs_per_worker=procs_per_worker)
        _result0 = context_monkeys.interface_monkeys.get('context_monkeys.interface_monkeys.global_rank')
        if context_monkeys.interface_monkeys.global_rank == 0:
            print _result0
        time.sleep(0.1)
    print ': context_monkeys.interface_monkeys.apply(init_worker)'
    print context_monkeys.interface_monkeys.apply(init_worker)
    print ': context_monkeys.interface_monkeys.apply(test, 1, 2, third=3)'
    print context_monkeys.interface_monkeys.apply(test, 1, 2, third=3)
    print ': context_monkeys.interface_monkeys.get(\'context_monkeys.first\')'
    print context_monkeys.interface_monkeys.get('context_monkeys.first')
    print ': context_monkeys.interface_monkeys.apply(test, 3, 4)'
    print context_monkeys.interface_monkeys.apply(test, 3, 4)
    print ': context_monkeys.interface_monkeys.get(\'context_monkeys.first\')'
    print context_monkeys.interface_monkeys.get('context_monkeys.first')
    context_monkeys.interface_monkeys.stop()


if __name__ == '__main__':
    main(standalone_mode=False)