from function_lib import *

context = Context()


def config_world(context, client_id):
    """
    """
    # requires a global variable context: :class:'Context'
    context.client_id = client_id


def collect_ids(tag, wait=None):
    """

    :param tag: int
    :return: tuple: int, dict
    """
    start_time = time.time()
    set_count()
    if wait is not None:
        time.sleep(wait)
    return context.client_id, {'tag': int(tag), 'compute time': time.time() - start_time, 'count': context.count}


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
    print 'client_id: %i, count: %i' % (context.client_id, context.count)


def main(cluster_id, profile):
    """

    :param cluster_id: str
    :param profile: str
    """
    if cluster_id is not None:
        c = Client(cluster_id=cluster_id, profile=profile)
    else:
        c = Client(profile=profile)
    context.c = c
    context.c[:].execute('from %s import *' % script_filename.split('.py')[0])
    context.c[:].map_sync(initialize_ipyp_engines, xrange(len(c)))
    view = c.load_balanced_view()
    results = []
    for i in xrange(20):
        results.append(view.map_async(collect_ids, [i]))  # , [0.1]))
    start_time = time.time()
    while not all([result.ready() for result in results]):
        pass
    print 'Took %.2f s' % (time.time() - start_time)
    for result in results:
        flush_engine_buffer(result)
        pprint.pprint(result.get())
        sys.stdout.flush()
