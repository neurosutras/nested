from parallel import *
import click


context_monkeys = Context()


def test(first, second, third=None):
    pid = os.getpid()
    print pid, first, second, third
    time.sleep(0.3)
    return pid, first, second, third


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
        context_monkeys.interface_monkeys = IpypInterface(cluster_id=cluster_id, profile=profile, procs_per_worker=procs_per_worker)
    elif framework == 'pc':
        context_monkeys.interface_monkeys = ParallelContextInterface(procs_per_worker=procs_per_worker)

    context_monkeys.interface_monkeys.start(disp=True)
    num_workers = context_monkeys.interface_monkeys.num_workers
    # result = context_monkeys.interface_monkeys.apply_get(test, 1, 2, 3)
    result1 = context_monkeys.interface_monkeys.apply(test, 1, 2, third=3)
    print result1
    result2 = context_monkeys.interface_monkeys.apply(test, 1, 2)
    print result2
    context_monkeys.interface_monkeys.stop()


if __name__ == '__main__':
    main()