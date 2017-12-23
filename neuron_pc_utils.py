from function_lib import *
from neuron import h
import click
import uuid


script_filename = 'neuron_pc_utils.py'

context = Context()
context.global_comm = MPI.COMM_WORLD


def initialize_subworlds(subworld_size=1):
    """
    neuron.h.ParallelContext provides a bulletin-board-style interface for concurrent processing, while also allowing
    collective communication within groups of processes (subworlds). While ParallelContext uses MPI for communication
    between processes, it does not expose the underlying MPI communicators for each subworld. This method initializes a
    global Context on each MPI rank to make both a split ParallelContext and a split MPI.COMM_WORLD accessible for
    nested parallel operations.
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
    # dictionary acts as a temporary storage container on the master process for results retrieved from the
    # ParallelContext bulletin board.
    context.collected = {}
    assert context.rank == context.comm.rank and context.global_rank == context.global_comm.rank and \
           context.global_comm.size / context.subworld_size == context.num_subworlds, 'pc.ids do not match MPI ranks'


def _pc_apply(func, key, *args):
    """
    Execute the specified function within a ParallelContext subworld. Block each subworld from returning until all
    subworlds have executed the function.
    :param func: callable
    :param key: string
    :param args: list
    """
    func(*args)
    if context.rank == 0:
        context.pc.take(key)
        count = context.pc.upkscalar()
        context.pc.post(key, count + 1)
        while True:
            if context.pc.look(key) and context.pc.upkscalar() == context.num_subworlds:
                return


def pc_apply(func, *args):
    """
    ParallelContext lacks a native method to guarantee execution of a function within all subworlds. This method
    implements a synchronous (blocking) apply operation.
    :param func: callable
    :param args: list
    """
    # key = str(uuid.uuid4())
    key = 'apply_'+func.__name__
    context.pc.post(key, 0)
    for i in xrange(context.num_subworlds):
        context.pc.submit(_pc_apply, func, key, *args)
    while context.pc.working():
        pass
    context.pc.take(key)


def pc_get_results(keys=None):
    """
    If no keys are specified, this method is a blocking operation that waits until all previously submitted jobs have
    been completed, retrieves all results from the bulletin board, and stores them in the 'collected' dict in the global
    Context on the master process, indexed by their submission key.
    If a list of keys is provided, pc_get_results first checks if the results have already been placed in the
    'collected' dict, and otherwise blocks until all requested results are available. Results retrieved from the
    bulletin board that were not requested are left in the 'collected' dict in the global Context.
    :param keys: list
    :return: dict
    """
    if keys is None:
        while context.pc.working():
            key = context.pc.userid()
            context.collected[key] = context.pc.pyret()
        keys = context.collected.keys()
        return {key: context.collected.pop(key) for key in keys}
    else:
        pending_keys = [key for key in keys if key not in context.collected]
        while pending_keys:
            if context.pc.working():
                key = context.pc.userid()
                context.collected[key] = context.pc.pyret()
                if key in pending_keys:
                    pending_keys.pop(key)
            else:
                break
        return {key: context.collected.pop(key) for key in keys if key in context.collected}


class PCAsyncResult(object):
    """
    Makes use of the 'collected' dict in the global Context on the master process to provide a non-blocking interface
    for collecting results from unfinished jobs.
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
        else:
            return True
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
            try:
                return [context.collected.pop(key) for key in self.keys]
            except KeyError:
                raise KeyError('PCAsyncResult: all jobs have completed, but not all requested keys were found')


def pc_map_sync(func, *sequences):
    """
    ParallelContext lacks a native method to apply a function to sequences of arguments, using all available
    processes, and returning the results in the same order as the specified sequence. This method implements a
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
    ParallelContext lacks a native method to apply a function to sequences of arguments, using all available
    processes, and returning the results in the same order as the specified sequence. This method implements an
    asynchronous (non-blocking) map operation. Returns a PCAsyncResult object to track progress of the submitted jobs.
    When ready(), get() returns results as a list in the same order as the specified sequences.
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
    while not results2.ready():
        pass
    results2 = results2.get()
    pprint.pprint(results2)
    print 'collected result keys: %s' % str(context.collected.keys())
    results4 = pc_get_results()
    pprint.pprint(results4)
    print 'collected result keys: %s' % str(context.collected.keys())
    while not results3.ready():
        pass
    results3 = results3.get()
    pprint.pprint(results3)
    print 'collected result keys: %s' % str(context.collected.keys())
    """
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
    """
    context.pc.done()
    h.quit()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_filename) != -1, sys.argv) + 1):])