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


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True,))
@click.option("--interactive", is_flag=True)
@click.pass_context
def main(cli, interactive):
    """

    :param cli: :class:'click.Context': used to process/pass through unknown click arguments
    :param interactive: bool
    """
    kwargs = get_unknown_click_arg_dict(cli.args)
    context.interface = get_parallel_interface(source_file=__file__, source_package=__package__, **kwargs)
    if 'framework' in kwargs:
        if kwargs['framework'] == 'ipyp':
            print('IpypInterface: before interface start: %i total processes detected' % context.interface.global_size)
            try:
                result1 = context.interface.get('MPI.COMM_WORLD.size')
                print('IpypInterface: ipengines each see an MPI.COMM_WORLD with size: %i' % max(result1))
            except RuntimeError:
                print('IpypInterface: ipengines do not see an MPI.COMM_WORLD')
        elif kwargs['framework'] == 'pc':
            result1 = context.interface.get('context.interface.global_rank')
            if context.interface.global_rank == 0:
                print('ParallelContextInterface: before interface start: %i / %i total processes participated in get '
                      'operation' % (len(set(result1)), context.interface.global_size))
        elif kwargs['framework'] == 'mpi':
            result1 = context.interface.get('context.global_comm.rank')
            print('MPIFuturesInterface: before interface start: %i / %i workers participated in get operation' %
                  (len(set(result1)), context.interface.num_workers))
        elif kwargs['framework'] == 'serial':
            result1 = context.interface.get('context.interface.num_workers')
            print('SerialInterface: before interface start: %i / %i workers participated in get operation' %
                  (len(set(result1)), context.interface.num_workers))
    sys.stdout.flush()
    time.sleep(1.)
    context.interface.start(disp=True)
    context.interface.ensure_controller()

    result2 = context.interface.apply(init_worker)
    sys.stdout.flush()
    time.sleep(1.)
    print('after interface start: %i / %i workers participated in apply(init_worker)\n' % \
          (len(set(result2)), context.interface.num_workers))
    sys.stdout.flush()
    time.sleep(1.)

    time_stamp = time.time()
    start1 = 0
    end1 = start1 + int(context.interface.global_size)
    start2 = end1
    end2 = start2 + int(context.interface.global_size)
    print(': context.interface.map_sync(test, range(%i, %i), range(%i, %i))' % (start1, end1, start2, end2))
    pprint.pprint(context.interface.map_sync(test, list(range(start1, end1)), list(range(start2, end2))))
    print('\n: map_sync took %.1f s\n' % (time.time() - time_stamp))
    sys.stdout.flush()
    time.sleep(1.)

    time_stamp = time.time()
    print(': context.interface.map_async(test, range(%i, %i), range(%i, %i))' % (start1, end1, start2, end2))
    result3 = context.interface.map_async(test, list(range(start1, end1)), list(range(start2, end2)))
    while not result3.ready(wait=0.1):
        pass
    result3 = result3.get()
    pprint.pprint(result3)
    sys.stdout.flush()
    print('\n: map_async took %.1f s\n' % (time.time() - time_stamp))
    sys.stdout.flush()
    time.sleep(1.)

    time_stamp = time.time()
    print(': context.interface.apply(test, 1, 2, third=3)')
    pprint.pprint(context.interface.apply(test, 1, 2, third=3))
    print('\n: apply took %.1f s\n' % (time.time() - time_stamp))
    sys.stdout.flush()
    time.sleep(1.)

    time_stamp = time.time()
    print(': context.interface.execute(init_worker)')
    result5 = context.interface.execute(init_worker)
    print('\n: execute returned: %s; took %.1f s\n' % (str(result5), time.time() - time_stamp))
    sys.stdout.flush()
    time.sleep(1.)

    time_stamp = time.time()
    print(': context.interface.get(\'context.pid\')')
    result4 = context.interface.get('context.pid')
    print('\n: get took %.1f s\n' % (time.time() - time_stamp))
    sys.stdout.flush()
    time.sleep(1.)
    print('before interface stop: %i / %i workers participated in get operation\n' % \
          (len(set(result4)), context.interface.num_workers))
    sys.stdout.flush()
    time.sleep(1.)

    if not interactive:
        context.interface.stop()


if __name__ == '__main__':
    main(standalone_mode=False)
