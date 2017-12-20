from function_lib import *
from neuron import h
import click
import uuid


script_filename = 'init_neuron.py'

context = Context()
context.global_comm = MPI.COMM_WORLD


def initialize_subworlds(subworld_size=1):
    """
    neuron.h.ParallelContext provides a bulletin-board style interface for concurrent processing, while also allowing
    collective communication within groups of processes (subworlds). While MPI is used for communication,
    ParallelContext does not expose the underlying MPI communicators for each subworld. This method initializes this
    parallelization scheme, and places both a split ParallelContext and a split MPI.COMM_WORLD into the global Context
    contained within each MPI rank.
    :param subworld_size: int
    """
    # requires a global variable context: :class:'Context'
    context.subworld_size = subworld_size
    context.pc = h.ParallelContext()
    context.pc.subworlds(subworld_size)
    context.global_rank = int(context.pc.id_world())
    context.global_size = int(context.pc.nhost_world())
    context.rank = int(context.pc.id())
    context.size = int(context.pc.nhost())
    global_ranks = [context.global_rank] * context.size
    global_ranks = context.pc.py_alltoall(global_ranks)
    group = context.global_comm.Get_group()
    sub_group = group.Incl(global_ranks)
    context.comm = context.global_comm.Create(sub_group)
    context.subworld_id = context.comm.bcast(int(context.pc.id_bbs()), root=0)
    context.num_subworlds = context.comm.bcast(int(context.pc.nhost_bbs()), root=0)
    context.collected = {}
    assert context.rank == context.comm.rank and context.global_rank == context.global_comm.rank and \
           context.global_comm.size / context.subworld_size == context.num_subworlds, 'pc.ids do not match MPI ranks'


def _pc_apply(func, key, *args):
    """
    After applying the specified function, keep each subworld from returning until all subworlds have applied as well.
    :param func: callable
    :param key: string
    :param args: list
    """
    func(*args)
    if context.rank == 0:
        context.pc.take(key)
        count = context.pc.upkscalar()
        context.pc.post(key, count + 1)
        while context.pc.look(key) and context.pc.upkscalar() < context.num_subworlds:
            pass
    return


def pc_apply(func, *args):
    """
    neuron.h.ParallelContext lacks a native method to apply a function to all ranks of all subworlds. This method makes
    use of a Context that contains a ParallelContext and an MPI.COMM_WORLD to implement an apply operation.
    :param func: callable
    :param args: list
    """
    key = str(uuid.uuid4())
    context.pc.post(key, 0)
    for i in xrange(context.num_subworlds):
        context.pc.submit(_pc_apply, func, key, *args)
    while context.pc.working():
        pass
    context.pc.take(key)


def pc_get_results(keys=None):
    """
    This method clears the neuron.h.ParallelContext bulletin board, and returns a dictionary of all results indexed by
    their submission key. If a list of keys are provided, pc_get_results returns only those requested results, leaving
    the rest of the cleared results in the dict 'collected' in the global Context.
    :param keys: list
    :return: dict
    """
    while context.pc.working():
        key = context.pc.userid()
        context.collected[key] = context.pc.pyret()
    if keys is None:
        keys = context.collected.keys()
    return {key: context.collected.pop(key) for key in keys if key in context.collected}


class PCAsyncResult(object):
    """
    Makes use of the dict 'collected' in the global Context to provide a non-blocking interface for collecting results
    from unfinished jobs.
    """
    def __init__(self, keys):
        """

        :param keys: list
        """
        self.keys = keys
        self._ready = False

    def ready(self):
        """

        :return: bool
        """
        if context.pc.working():
            key = context.pc.userid()
            context.collected[key] = context.pc.pyret()
        if all(key in context.collected for key in self.keys):
            return True
        else:
            return False

    def get(self):
        """
        Returns None until all results have completed, then returns a list of results in the order of original
        submission.
        :return: list
        """
        if not self.ready():
            return None
        else:
            return [context.collected.pop(key) for key in self.keys]


def pc_map_sync(func, *sequences):
    """
    neuron.h.ParallelContext lacks a native method to map a sequence of parameters to a function, using all available
    subworlds. This method makes use of a Context that contains a ParallelContext and an MPI.COMM_WORLD to implement a
    synchronous (blocking) map operation. Returns results as a list in the same order as the specified sequences.
    :param func: callable
    :param sequences: list
    :return: list
    """
    if not sequences:
        return None
    keys = []
    for args in zip(*sequences):
        key = abs(context.pc.submit(func, *args))
        keys.append(key)
    results = PCAsyncResult(keys)
    while not results.ready():
        pass
    return results.get()


def pc_map_async(func, *sequences):
    """
    neuron.h.ParallelContext lacks a native method to map a sequence of parameters to a function, using all available
    subworlds. This method makes use of a Context that contains a ParallelContext and an MPI.COMM_WORLD to implement an
    asynchronous (blocking) map operation. When ready(), get() returns results as a list in the same order as the
    specified sequences.
    :param func: callable
    :param sequences: list
    :return: list
    """
    if not sequences:
        return None
    keys = []
    for args in zip(*sequences):
        key = abs(context.pc.submit(func, *args))
        keys.append(key)
    return PCAsyncResult(keys)


def collect_ranks(tag):
    """

    :param tag: int
    :return: tuple: int, dict
    """
    start_time = time.time()
    ranks = context.comm.gather(context.global_rank, root=0)
    if context.rank == 0:
        return context.subworld_id, {'ranks': ranks, 'tag': int(tag), 'compute time': time.time() - start_time}


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
          (context.global_rank, context.global_size, context.rank, context.size,
           context.subworld_id, context.num_subworlds, context.count)


@click.command()
@click.option("--subworld-size", type=int, default=1)
def main(subworld_size):
    """

    :param subworld_size: int
    """
    initialize_subworlds(subworld_size)
    context.pc.runworker()
    pc_apply(set_count)
    pc_apply(set_count, 5)
    results1 = pc_map_sync(collect_ranks, range(10))
    pprint.pprint(results1)
    results2 = pc_map_async(collect_ranks, range(10, 20))
    results3 = pc_map_async(collect_ranks, range(20, 30))
    start_time = time.time()
    print 'collected result keys: %s' % str(context.collected.keys())
    while not (results2.ready() and results3.ready()):
        pass
    print 'Could have been working on rank %i! Elapsed time: %.3f' % (context.global_rank, time.time() - start_time)
    print 'collected result keys: %s' % str(context.collected.keys())
    results2 = results2.get()
    pprint.pprint(results2)
    print 'collected result keys: %s' % str(context.collected.keys())
    results3 = results3.get()
    pprint.pprint(results3)
    print 'collected result keys: %s' % str(context.collected.keys())
    context.pc.done()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_filename) != -1, sys.argv) + 1):])