from utils import *


class IpypInterface(object):
    """

    """

    class AsyncResultWrapper(object):
        """

        """

        def __init__(self, async_result):
            self.async_result = async_result
            self._ready = False
            self.stdout = []

        def ready(self):
            if not self._ready:
                self.stdout = self.async_result.stdout
            self._ready = self.async_result.ready()
            return self._ready

        def get(self):
            if self.ready():
                self.stdout_flush()
                return self.async_result.get()
            else:
                return None

        def stdout_flush(self):
            """
            Once an async_result is ready, print the contents of its stdout buffer.
            :param result: :class:'ASyncResult
            """
            for stdout in self.stdout:
                if stdout:
                    for line in stdout.splitlines():
                        print line
            sys.stdout.flush()

    def __init__(self, cluster_id=None, profile='default', procs_per_worker=1, sleep=0):
        """
        :param cluster_id: str
        :param profile: str
        :param procs_per_worker: int
        :param sleep: int
        TODO: Implement nested collective operations for ipyp interface
        """
        try:
            from ipyparallel import Client
        except ImportError:
            raise ImportError('nested: ipyp framework: problem with importing ipyparallel')
        if cluster_id is not None:
            self.client = Client(cluster_id=cluster_id, profile=profile)
        else:
            self.client = Client(profile=profile)
        self.global_size = len(self.client)
        if procs_per_worker > 1:
            print 'nested: ipyp framework: procs_per_worker reduced to 1; collective operations not yet implemented'
        self.num_procs_per_worker = 1
        self.num_workers = self.global_size / self.num_procs_per_worker
        self.direct_view = self.client
        self.load_balanced_view = self.client.load_balanced_view()
        self.direct_view[:].execute('from nested.optimize import *', block=True)
        time.sleep(sleep)
        self.apply_sync = \
            lambda func, *args, **kwargs: \
                self._sync_wrapper(self.AsyncResultWrapper(self.direct_view[:].apply_async(func, *args, **kwargs)))
        """
        self.apply_async = lambda func, *args, **kwargs: \
            self.AsyncResultWrapper(self.direct_view[:].apply_async(func, *args, **kwargs))
        """
        self.apply = self.apply_sync
        self.map_sync = \
            lambda func, *args: self._sync_wrapper(self.AsyncResultWrapper(self.direct_view[:].map_async(func, *args)))
        self.map_async = lambda func, *args: self.AsyncResultWrapper(self.load_balanced_view.map_async(func, *args))
        self.map = self.map_sync
        self.get = lambda context, x: [getattr(item, x) for item in self.direct_view[:][context]]

    def _sync_wrapper(self, async_result_wrapper):
        """

        :param async_result_wrapper: :class:'ASyncResultWrapper'
        :return: list
        """
        while not async_result_wrapper.ready():
            pass
        return async_result_wrapper.get()

    def print_info(self):
        print 'IpypInterface: process id: %i; num workers: %i' % (os.getpid(), self.num_workers)
        sys.stdout.flush()

    def start(self, disp=False):
        if disp:
            self.print_info()
        return

    def stop(self):
        return


class ParallelContextInterface(object):
    """

    """
    class AsyncResultWrapper(object):
        """
        When ready(), get() returns results as a list in the same order as submission.
        """
        def __init__(self, interface, keys):
            """

            :param interface: :class: 'ParallelContextInterface'
            :param keys: list
            """
            self.interface = interface
            self.keys = keys
            self._ready = False

        def ready(self):
            """

            :return: bool
            """
            if self.interface.pc.working():
                key = int(self.interface.pc.userid())
                self.interface.collected[key] = self.interface.pc.pyret()
            else:
                self._ready = True
                return True
            if all(key in self.interface.collected for key in self.keys):
                self._ready = True
                return True
            else:
                return False

        def get(self):
            """
            Returns None until all results have completed, then returns a list of results in the order of original
            submission.
            :return: list
            """
            if self._ready or self.ready():
                try:
                    return [self.interface.collected.pop(key) for key in self.keys]
                except KeyError:
                    raise KeyError('nested: neuron.h.ParallelContext framework: AsyncResultWrapper: all jobs have '
                                   'completed, but not all requested keys were found')
            else:
                return None
    
    def __init__(self, procs_per_worker=1):
        """

        :param procs_per_worker: int
        """
        try:
            from mpi4py import MPI
            from neuron import h
        except ImportError:
            raise ImportError('nested: neuron.h.ParallelContext framework: problem with importing neuron')
        self.global_comm = MPI.COMM_WORLD
        self.procs_per_worker = procs_per_worker
        self.pc = h.ParallelContext()
        self.pc.subworlds(procs_per_worker)
        self.global_rank = int(self.pc.id_world())
        self.global_size = int(self.pc.nhost_world())
        self.rank = int(self.pc.id())
        self.size = int(self.pc.nhost())
        global_ranks = [self.global_rank] * self.size
        global_ranks = self.pc.py_alltoall(global_ranks)
        group = self.global_comm.Get_group()
        sub_group = group.Incl(global_ranks)
        self.comm = self.global_comm.Create(sub_group)
        self.worker_id = self.comm.bcast(int(self.pc.id_bbs()), root=0)
        self.num_workers = self.comm.bcast(int(self.pc.nhost_bbs()), root=0)
        # 'collected' dict acts as a temporary storage container on the master process for results retrieved from
        # the ParallelContext bulletin board.
        self.collected = {}
        assert self.rank == self.comm.rank and self.global_rank == self.global_comm.rank and \
               self.global_comm.size / self.procs_per_worker == self.num_workers, \
            'nested: neuron.h.ParallelContext framework: pc.ids do not match MPI ranks'
        self._running = False
        self.map = self.map_sync
        self.apply = self.apply_sync
        self.apply_counter = 0

    def print_info(self):
        print 'ParallelContextInterface: process id: %i; global rank: %i / %i; local rank: %i / %i; ' \
              'worker id: %i / %i' % \
              (os.getpid(), self.global_rank, self.global_size, self.comm.rank, self.comm.size, self.worker_id,
               self.num_workers)
        # time.sleep(0.1)

    def _apply(self, func, *args, **kwargs):
        """
        ParallelContext lacks a native method to guarantee execution of a function on all workers. This method
        implements a synchronous (blocking) apply operation that accepts **kwargs and returns values collected from each
        worker
        :param func: callable
        :param args: list
        :param kwargs: dict
        :return: dynamic
        """
        apply_key = self.apply_counter
        self.apply_counter += 1
        self.pc.post(apply_key, 0)
        keys = []
        for i in xrange(self.num_workers):
            keys.append(int(self.pc.submit(pc_apply_wrapper, func, apply_key, args, kwargs)))
        results = self.collect_results(keys)
        self.pc.take(apply_key)
        return [results[key] for key in keys]

    def apply_sync(self, func, *args, **kwargs):
        """
        ParallelContext lacks a native method to guarantee execution of a function on all workers. This method
        implements a synchronous (blocking) apply operation that accepts **kwargs and returns values collected from each
        worker.
        :param func: callable
        :param args: list
        :param kwargs: dict
        :return: dynamic
        """
        if self._running:
            return self._apply(func, *args, **kwargs)
        else:
            result = func(*args, **kwargs)
            if not self._running:
                results = self.global_comm.gather(result, root=0)
                if self.global_rank == 0:
                    return results
            else:
                return [result]
    
    def collect_results(self, keys=None):
        """
        If no keys are specified, this method is a blocking operation that waits until all previously submitted jobs 
        have been completed, retrieves all results from the bulletin board, and stores them in the 'collected' dict in 
        on the master process, indexed by their submission key.
        If a list of keys is provided, collect_results first checks if the results have already been placed in the
        'collected' dict, and otherwise blocks until all requested results are available. Results retrieved from the
        bulletin board that were not requested are left in the 'collected' dict.
        :param keys: list
        :return: dict
        """
        if keys is None:
            while self.pc.working():
                key = int(self.pc.userid())
                self.collected[key] = self.pc.pyret()
            keys = self.collected.keys()
            return {key: self.collected.pop(key) for key in keys}
        else:
            pending_keys = [key for key in keys if key not in self.collected]
            while pending_keys:
                if self.pc.working():
                    key = int(self.pc.userid())
                    self.collected[key] = self.pc.pyret()
                    if key in pending_keys:
                        pending_keys.remove(key)
                else:
                    break
            return {key: self.collected.pop(key) for key in keys if key in self.collected}

    def map_sync(self, func, *sequences):
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
            key = int(self.pc.submit(func, *args))
            keys.append(key)
        results = self.collect_results(keys)
        return [results[key] for key in keys]

    def map_async(self, func, *sequences):
        """
        ParallelContext lacks a native method to apply a function to sequences of arguments, using all available
        processes, and returning the results in the same order as the specified sequence. This method implements an
        asynchronous (non-blocking) map operation. Returns a PCAsyncResult object to track progress of the submitted
        jobs.
        :param func: callable
        :param sequences: list
        :return: list
        """
        if not sequences:
            return None
        keys = []
        for args in zip(*sequences):
            key = int(self.pc.submit(func, *args))
            keys.append(key)
        return self.AsyncResultWrapper(self, keys)

    def start(self, disp=False):
        if disp:
            self.print_info()
            # time.sleep(0.1)
        self._running = True
        self.pc.runworker()

    def stop(self):
        self.pc.done()
        self._running = False


def pc_apply_wrapper(func, key, args, kwargs):
    """

    :param func: callable
    :param key: str
    :param args: list
    :param kwargs: dict
    :return: dynamic
    """
    result = func(*args, **kwargs)
    interface = None
    try:
        module = sys.modules['__main__']
        for item_name in dir(module):
            if isinstance(getattr(module, item_name), ParallelContextInterface):
                interface = getattr(module, item_name)
                break
        if interface is None:
            context = None
            for item_name in dir(module):
                if isinstance(getattr(module, item_name), Context):
                    context = getattr(module, item_name)
                    break
            if context is not None:
                for item_name in context():
                    if isinstance(getattr(context, item_name), ParallelContextInterface):
                        interface = getattr(context, item_name)
                        break
            if interface is None:
                raise Exception
    except Exception:
        raise Exception('ParallelContextInterface: required objects not found in __main__ namespace: '
                        'context: :class:\'Context\' and context.interface: :class:\'ParallelContextInterface\'')
    if interface.rank == 0:
        interface.pc.take(key)
        count = interface.pc.upkscalar()
        interface.pc.post(key, count + 1)
        while True:
            if interface.pc.look(key) and interface.pc.upkscalar() == interface.num_workers:
                return result
