"""
Nested parallel processing tools.

Classes and methods to provide a consistent interface for various parallel processing frameworks.
Used by nested.optimize
"""
__author__ = 'Aaron D. Milstein'
from nested.utils import *


class IpypInterface(object):
    """

    """

    class AsyncResultWrapper(object):
        """

        """

        def __init__(self, interface, async_result):
            """
            :param async_result: :class:'ASyncResult'
            """
            self.interface = interface
            self.async_result = async_result
            self._ready = False
            self.stdout = []

        def ready(self, wait=None):
            """

            :param wait: int or float
            :return: bool
            """
            try:
                if not self._ready:
                    self.stdout = self.async_result.stdout
                else:
                    return True
                self._ready = self.async_result.ready()
            except Exception:
                traceback.print_exc(file=sys.stdout)
                self.interface.hard_stop()
            if not self._ready and wait is not None:
                time.sleep(wait)
            return self._ready

        def get(self):
            if self.ready():
                self.stdout_flush()
                try:
                    result = self.async_result.get()
                except Exception:
                    traceback.print_exc(file=sys.stdout)
                    self.interface.hard_stop()
                return result
            else:
                return None

        def stdout_flush(self):
            """
            Once an async_result is ready, print the contents of its stdout buffer.
            """
            for stdout in self.stdout:
                if stdout:
                    for line in stdout.splitlines():
                        print(line)
            sys.stdout.flush()

    def __init__(self, cluster_id=None, profile='default', procs_per_worker=1, sleep=0, source_file=None,
                 source_package=None):
        """
        Instantiates an interface to an ipyparallel.Client on the master process. Imports the calling source script on
        all available workers (ipengines).
        :param cluster_id: str
        :param profile: str
        :param procs_per_worker: int
        :param sleep: int   # dv.execute fails to block on some clusters. Allow engines time to import modules.
        :param source_file: str
        :param source_package: str
        """
        try:
            from ipyparallel import Client
        except ImportError:
            raise ImportError('nested: IpypInterface: problem with importing ipyparallel')
        if cluster_id is not None:
            self.client = Client(cluster_id=cluster_id, profile=profile)
        else:
            self.client = Client(profile=profile)
        self.global_size = len(self.client)
        if procs_per_worker > 1:
            print('nested: IpypInterface: procs_per_worker reduced to 1; collective operations not yet implemented')
        self.procs_per_worker = 1
        self.num_workers = int(self.global_size / self.procs_per_worker)
        self.direct_view = self.client
        self.load_balanced_view = self.client.load_balanced_view()
        if source_file is None:
            source_file = sys.argv[0]
        source_dir = os.path.dirname(os.path.abspath(source_file))
        sys.path.insert(0, source_dir)
        if source_package is not None:
            source = source_package + '.'
        else:
            source = ''
        source += os.path.basename(source_file).split('.py')[0]
        try:
            self.direct_view[:].execute('from %s import *' % source, block=True)
            time.sleep(sleep)
        except Exception:
            raise Exception('nested.parallel: IPypInterface: failed to import source: %s from dir: %s' %
                            (source, source_dir))
        self.apply_sync = \
            lambda func, *args, **kwargs: \
                self._sync_wrapper(self.AsyncResultWrapper(self, self.direct_view[:].apply_async(
                    parallel_execute_wrapper, func, args, kwargs)))
        self.apply = self.apply_sync
        self.execute = \
            lambda func, *args, **kwargs: \
                self._sync_wrapper(self.AsyncResultWrapper(self, self.direct_view[0].apply_async(
                    parallel_execute_wrapper, func, args, kwargs)))
        self.map = self.map_sync
        self.get = lambda x: self.direct_view[:][x]
        self.apply(ipyp_init_workers, num_workers=self.num_workers)
        self.controller_is_worker = False
        self.print_info()

    def _sync_wrapper(self, async_result_wrapper):
        """

        :param async_result_wrapper: :class:'ASyncResultWrapper'
        :return: list
        """
        while not async_result_wrapper.ready():
            time.sleep(0.3)
        return async_result_wrapper.get()

    def map_sync(self, func, *args):
        group_size = len(args)
        sequences = zip(*args)
        return self._sync_wrapper(self.AsyncResultWrapper(self, self.direct_view[:].map_async(
            parallel_execute_wrapper, [func] * group_size, sequences)))

    def map_async(self, func, *args):
        group_size = len(args)
        sequences = zip(*args)
        return self.AsyncResultWrapper(self, self.load_balanced_view.map_async(
            parallel_execute_wrapper, [func] * group_size, sequences))

    def print_info(self):
        print('nested: IpypInterface: process id: %i; num workers: %i' % (os.getpid(), self.num_workers))
        sys.stdout.flush()

    def update_worker_contexts(self, content=None, **kwargs):
        """
        Data provided either through the positional argument content as a dictionary, or through kwargs, will be used to
        update the remote Context objects found on all workers, using an apply operation.
        :param content: dict
        """
        if content is None:
            content = dict()
        content.update(kwargs)
        self.apply(update_worker_contexts, content)

    def start(self, disp=False):
        pass

    def stop(self):
        os._exit(1)

    def hard_stop(self):
        print('nested: IpypInterface: an Exception on a worker process brought down the whole operation')
        sys.stdout.flush()
        time.sleep(1.)
        os._exit(1)

    def ensure_controller(self):
        pass


def ipyp_init_workers(**content):
    """
    Push a content dictionary into the local context on each engine.
    :param context: dict
    """
    local_context = find_context()
    local_context.update(content)


class MPIFuturesInterface(object):
    """
    Class provides an interface to extend the mpi4py.futures concurrency tools for flexible nested parallel
    computations.
    """

    class AsyncResultWrapper(object):
        """
        When ready(), get() returns results as a list in the same order as submission.
        """

        def __init__(self, interface, futures):
            """

            :param futures: list of :class:'mpi4py.futures.Future'
            """
            self.interface = interface
            self.futures = futures
            self._ready = False

        def ready(self, wait=None):
            """
            :param wait: int or float
            :return: bool
            """
            time_stamp = time.time()
            if wait is None:
                wait = 0
            try:
                while not np.all([future.done() for future in self.futures]):
                    if time.time() - time_stamp > wait:
                        return False
            except Exception:
                traceback.print_exc(file=sys.stdout)
                self.interface.hard_stop()
            self._ready = True
            return True

        def get(self):
            """
            Returns None until all results have completed, then returns a list of results in the order of original
            submission.
            :return: list
            """
            if self._ready or self.ready():
                try:
                    results = [future.result() for future in self.futures]
                except Exception:
                    traceback.print_exc(file=sys.stdout)
                    self.interface.hard_stop()
                return results
            else:
                return None

    def __init__(self, procs_per_worker=1):
        """

        :param procs_per_worker: int
        """
        try:
            from mpi4py import MPI
            from mpi4py.futures import MPIPoolExecutor
        except ImportError:
            raise ImportError('nested: MPIFuturesInterface: problem with importing from mpi4py.futures')
        self.global_comm = MPI.COMM_WORLD
        if procs_per_worker > 1:
            print('nested: MPIFuturesInterface: procs_per_worker reduced to 1; collective operations not yet '
                  'implemented')
        self.procs_per_worker = 1
        self.executor = MPIPoolExecutor()
        self.rank = self.global_comm.rank
        self.global_size = self.global_comm.size
        self.num_workers = self.global_size - 1
        self.apply_counter = 0
        self.map = self.map_sync
        self.apply = self.apply_sync
        self.init_workers(disp=True)
        self.controller_is_worker = False

    def init_workers(self, disp=False):
        """

        :param disp: bool
        """
        futures = []
        for task_id in range(1, self.global_size):
            futures.append(self.executor.submit(mpi_futures_init_workers, task_id, disp))
        mpi_futures_init_workers(0)
        try:
            results = [future.result() for future in futures]
            num_returned = len(set(results))
            if num_returned != self.num_workers:
                raise ValueError('nested: MPIFuturesInterface: %i / %i processes returned from init_workers' %
                                 (num_returned, self.num_workers))
        except Exception:
            traceback.print_exc(file=sys.stdout)
            self.hard_stop()
        self.print_info()

    def print_info(self):
        """

        """
        print('nested: MPIFuturesInterface: process id: %i; rank: %i / %i; num_workers: %i' %
              (os.getpid(), self.rank, self.global_size, self.num_workers))
        sys.stdout.flush()
        time.sleep(0.1)

    def apply_sync(self, func, *args, **kwargs):
        """
        mpi4py.futures lacks a native method to guarantee execution of a function on all workers. This method
        implements a synchronous (blocking) apply operation that accepts **kwargs and returns values collected from each
        worker.
        :param func: callable
        :param args: list
        :param kwargs: dict
        :return: dynamic
        """
        apply_key = int(self.apply_counter)
        self.apply_counter += 1
        futures = []
        for rank in range(1, self.global_size):
            futures.append(self.executor.submit(mpi_futures_apply_wrapper, func, apply_key, args, kwargs))
        try:
            results = [future.result() for future in futures]
        except Exception:
            traceback.print_exc(file=sys.stdout)
            self.hard_stop()
        return results

    def execute(self, func, *args, **kwargs):
        """
        This method executes a function on a single worker and returns the result.
        :param func: callable
        :param args: list
        :param kwargs: dict
        :return: dynamic
        """
        future = self.executor.submit(parallel_execute_wrapper, func, args, kwargs)
        try:
            result = future.result()
        except Exception:
            traceback.print_exc(file=sys.stdout)
            self.hard_stop()
        return result

    def map_sync(self, func, *sequences):
        """
        This method wraps mpi4py.futures.MPIPoolExecutor.map to implement a synchronous (blocking) map operation.
        Uses all available processes, and returns results as a list in the same order as the specified sequences.
        :param func: callable
        :param sequences: list
        :return: list
        """
        if not sequences:
            return None
        futures = []
        for args in zip(*sequences):
            futures.append(self.executor.submit(parallel_execute_wrapper, func, args))
        try:
            results = [future.result() for future in futures]
        except Exception:
            traceback.print_exc(file=sys.stdout)
            self.hard_stop()
        return results

    def map_async(self, func, *sequences):
        """
        This method wraps mpi4py.futures.MPIPoolExecutor.submit to implement an asynchronous (non-blocking) map
        operation. Uses all available processes, and returns results as a list in the same order as the specified
        sequences. Returns an AsyncResultWrapper object to track progress of the submitted jobs.
        :param func: callable
        :param sequences: list
        :return: list
        """
        if not sequences:
            return None
        futures = []
        for args in zip(*sequences):
            futures.append(self.executor.submit(parallel_execute_wrapper, func, args))
        return self.AsyncResultWrapper(self, futures)

    def get(self, object_name):
        """
        mpi4py.futures lacks a native method to get the value of an object from all workers. This method implements a
        synchronous (blocking) pull operation.
        :param object_name: str
        :return: dynamic
        """
        return self.apply_sync(find_nested_object, object_name)

    def update_worker_contexts(self, content=None, **kwargs):
        """
        Data provided either through the positional argument content as a dictionary, or through kwargs, will be used to
        update the remote Context objects found on all workers, using an apply operation.
        :param content: dict
        """
        if content is None:
            content = dict()
        content.update(kwargs)
        self.apply(update_worker_contexts, content)

    def start(self, disp=False):
        pass

    def stop(self):
        self.executor.shutdown()

    def hard_stop(self):
        print('nested: MPIFuturesInterface: an Exception on a worker process brought down the whole operation')
        sys.stdout.flush()
        time.sleep(1.)
        self.executor.shutdown(wait=False)
        os._exit(1)

    def ensure_controller(self):
        """
        Exceptions in python on an MPI rank are not enough to end a job. Strange behavior results when an unhandled
        Exception occurs on an MPI rank while under the control of an mpi4py.futures.MPIPoolExecutor. This method will
        hard exit python if executed by any rank other than the master.
        """
        if self.rank != 0:
            os._exit(1)


def mpi_futures_wait_for_all_workers(comm, key, disp=False):
    """
    The master rank 0 is busy managing the executor. Any job submitted to the executor can be picked up by any worker
    process that is ready. This method forces all workers that pick up a job to wait for a handshake with rank 1 before
    starting work, thereby guaranteeing that each worker will participate in the operation.
    :param comm: :class:'MPI.COMM_WORLD'
    :param key: int
    :param disp: bool; verbose reporting for debugging
    """
    start_time = time.time()
    if comm.rank == 1:
        open_ranks = list(range(2, comm.size))
        for worker_rank in open_ranks:
            future = comm.irecv(source=worker_rank)
            val = future.wait()
            if val != worker_rank:
                raise ValueError('nested: MPIFuturesInterface: process id: %i; rank: %i; received wrong value: %i; '
                                 'from worker: %i' % (os.getpid(), comm.rank, val, worker_rank))
        for worker_rank in open_ranks:
            comm.isend(key, dest=worker_rank)
        if disp:
            print('Rank: %i took %.3f s to complete wait_for_all_workers' % (comm.rank, time.time() - start_time))
            sys.stdout.flush()
            time.sleep(0.1)
    else:
        comm.isend(comm.rank, dest=1)
        future = comm.irecv(source=1)
        val = future.wait()
        if val != key:
            raise ValueError('nested: MPIFuturesInterface: process id: %i; rank: %i; expected apply_key: '
                             '%i; received: %i from rank: 1' % (os.getpid(), comm.rank, key, val))


def mpi_futures_init_workers(task_id, disp=False):
    """
    Create an MPI communicator and insert it into a local Context object on each remote worker.
    :param task_id: int
    :param disp: bool
    """
    local_context = find_context()
    if 'global_comm' not in local_context():
        try:
            from mpi4py import MPI
        except ImportError:
            raise ImportError('nested: MPIFuturesInterface: problem with importing from mpi4py on workers')
        local_context.global_comm = MPI.COMM_WORLD
        local_context.num_workers = local_context.global_comm.size - 1
        local_context.comm = MPI.COMM_SELF
    if task_id != local_context.global_comm.rank:
        raise ValueError('nested: MPIFuturesInterface: mpi_futures_init_workers: process id: %i; rank: %i; '
                         'received wrong task_id: %i' % (os.getpid(), local_context.global_comm.rank, task_id))
    if disp:
        print('nested: MPIFuturesInterface: process id: %i; rank: %i / %i; procs_per_worker: %i' %
              (os.getpid(), local_context.global_comm.rank, local_context.global_comm.size, local_context.comm.size))
        sys.stdout.flush()
        time.sleep(0.1)
    return local_context.global_comm.rank


def update_worker_contexts(content):
    """
    nested.parallel interfaces require a remote instance of Context. This method can be used by an apply operation
    to update each remote Context with the contents of the provided content dictionary.
    :param content: dict
    """
    local_context = find_context()
    local_context.update(content)


def find_context():
    """
    nested.parallel interfaces require a remote instance of Context. This method attempts to find it in the remote
    __main__ namespace.
    :return: :class:'Context'
    """
    local_context = None
    try:
        module = sys.modules['__main__']
        for item_name in dir(module):
            if isinstance(getattr(module, item_name), Context):
                local_context = getattr(module, item_name)
                return local_context
        if local_context is None:
            raise Exception
    except Exception:
        raise Exception('nested.parallel: problem finding remote instance of Context in the remote namespace for '
                        'module: %s' % module.__name__)


def find_context_name(source=None):
    """
    nested.parallel interfaces require a remote instance of Context. This method attempts to find it in namespace of
    the provided module, and returns its string name.
    :param source: str; name of module
    :return: str
    """
    item_name = None
    try:
        if source is None:
            module = sys.modules['__main__']
        else:
            module = sys.modules[source]
        for item_name in dir(module):
            if isinstance(getattr(module, item_name), Context):
                return item_name
        if item_name is None:
            raise Exception
    except Exception:
        raise Exception('nested.parallel: problem finding remote instance of Context in the remote namespace for '
                        'module: %s' % source)


def mpi_futures_apply_wrapper(func, key, args, kwargs):
    """
    Method used by MPIFuturesInterface to implement an 'apply' operation. As long as a module executes
    'from nested.parallel import *', this method can be executed remotely, and prevents any worker from returning until
    all workers have applied the specified function.
    :param func: callable
    :param key: int
    :param args: list
    :param kwargs: dict
    :return: dynamic
    """
    local_context = find_context()
    mpi_futures_wait_for_all_workers(local_context.global_comm, key)
    result = parallel_execute_wrapper(func, args, kwargs)
    return result


def find_nested_object(object_name):
    """
    This method attempts to find the object corresponding to the provided object_name (str) in the __main__ namespace.
    Tolerates objects nested in other objects.
    :param object_name: str
    :return: dynamic
    """
    this_object = None
    try:
        module = sys.modules['__main__']
        for this_object_name in object_name.split('.'):
            if this_object is None:
                this_object = getattr(module, this_object_name)
            else:
                this_object = getattr(this_object, this_object_name)
        if this_object is None:
            raise Exception
        return this_object
    except Exception:
        raise Exception('nested: object: %s not found in remote __main__ namespace' % object_name)


class ParallelContextInterface(object):
    """
    Class provides an interface to extend the NEURON ParallelContext bulletin board for flexible nested parallel
    computations.
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
            self.remaining_keys = list(keys)
            self._ready = False

        def ready(self, wait=None):
            """
            :param wait: int or float
            :return: bool
            """
            time_stamp = time.time()
            if wait is None:
                wait = 0
            try:
                while len(self.remaining_keys) > 0 and self.interface.pc.working():
                    key = int(self.interface.pc.userid())
                    self.interface.collected[key] = self.interface.pc.pyret()
                    try:
                        self.remaining_keys.remove(key)
                    except ValueError:
                        pass
                    if time.time() - time_stamp > wait:
                        return False
            except Exception:
                traceback.print_exc(file=sys.stdout)
                self.interface.hard_stop()
            self._ready = True
            return True

        def get(self):
            """
            Returns None until all results have completed, then returns a list of results in the order of original
            submission.
            :return: list
            """
            if self._ready or self.ready():
                try:
                    return [self.interface.collected.pop(key) for key in self.keys]
                except Exception:
                    traceback.print_exc(file=sys.stdout)
                    self.interface.hard_stop()
            else:
                return None

    def __init__(self, procs_per_worker=1):
        """

        :param procs_per_worker: int
        """
        try:
            from mpi4py import MPI
            from neuron import h
        except Exception:
            raise ImportError('nested: ParallelContextInterface: problem with importing neuron')
        try:
            h.nrnmpi_init()
        except Exception:
            print('nested: ParallelContextInterface: h.nrnmpi_init() not executed; may not be defined in this version '
                  'of NEURON')
        self.global_comm = MPI.COMM_WORLD
        group = self.global_comm.Get_group()
        sub_group = group.Incl(list(range(1, self.global_comm.size)))
        self.worker_comm = self.global_comm.Create(sub_group)
        self.procs_per_worker = procs_per_worker
        self.h = h
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
               self.global_comm.size // self.procs_per_worker == self.num_workers, \
            'nested: ParallelContextInterface: pc.ids do not match MPI ranks'
        self._running = False
        self.map = self.map_sync
        self.apply = self.apply_sync
        self.key_counter = 0
        self.maxint = 1e7
        self.controller_is_worker = True

    def print_info(self):
        print('nested: ParallelContextInterface: process id: %i; global rank: %i / %i; local rank: %i / %i; '
              'worker id: %i / %i' %
              (os.getpid(), self.global_rank, self.global_size, self.comm.rank, self.comm.size, self.worker_id,
               self.num_workers))
        sys.stdout.flush()
        time.sleep(0.1)

    def get_next_key(self):
        """
        The ParallelContext bulletin board associates each job with an id, but it is limited to the size of a c++ int,
        so it must be reset occasionally.
        :return: int
        """
        if self.key_counter >= self.maxint:
            self.key_counter = 0
        key = self.key_counter
        self.key_counter += 1
        return key

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
            apply_key = int(self.get_next_key())
            keys = []
            for i in range(self.num_workers):
                key = int(self.get_next_key())
                self.pc.submit(key, pc_apply_wrapper, func, apply_key, args, kwargs)
                keys.append(key)
            results = self.collect_results(keys)
            sys.stdout.flush()
            return results
        else:
            result = parallel_execute_wrapper(func, args, kwargs)
            sys.stdout.flush()
            if not self._running:
                results = self.global_comm.gather(result, root=0)
                if self.global_rank == 0:
                    return results
            elif result is None:
                return
            else:
                return [result]

    def collect_results(self, keys=None):
        """
        If no keys are specified, this method is a blocking operation that waits until all previously submitted jobs 
        have been completed, retrieves all results from the bulletin board, and returns them as a dict indexed by their
        submission key.
        If a list of keys is provided, collect_results first checks if the results have already been placed in the
        'collected' dict on the master process, and otherwise blocks until all requested results are available. Results
        are returned as a list in the same order as the submitted keys. Results retrieved from the bulletin board that
        were not requested are left in the 'collected' dict.
        :param keys: list
        :return: list or dict
        """
        try:
            if keys is None:
                while self.pc.working():
                    key = int(self.pc.userid())
                    self.collected[key] = self.pc.pyret()
                keys = list(self.collected.keys())
                return {key: self.collected.pop(key) for key in keys}
            else:
                remaining_keys = [key for key in keys if key not in self.collected]
                while len(remaining_keys) > 0 and self.pc.working():
                    key = int(self.pc.userid())
                    self.collected[key] = self.pc.pyret()
                    try:
                        remaining_keys.remove(key)
                    except ValueError:
                        pass
                return [self.collected.pop(key) for key in keys]
        except Exception:
            traceback.print_exc(file=sys.stdout)
            self.hard_stop()

    def execute(self, func, *args, **kwargs):
        """
        This method executes a function on a single worker and returns the result.
        :param func: callable
        :param args: list
        :param kwargs: dict
        :return: dynamic
        """
        key = int(self.get_next_key())
        self.pc.submit(key, parallel_execute_wrapper, func, args, kwargs)
        result = self.collect_results([key])[0]
        sys.stdout.flush()
        return result

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
            key = int(self.get_next_key())
            self.pc.submit(key, parallel_execute_wrapper, func, args)
            keys.append(key)
        results = self.collect_results(keys)
        return results

    def map_async(self, func, *sequences):
        """
        ParallelContext lacks a native method to apply a function to sequences of arguments, using all available
        processes, and returning the results in the same order as the specified sequence. This method implements an
        asynchronous (non-blocking) map operation. Returns a AsyncResultWrapper object to track progress of the
        submitted jobs.
        :param func: callable
        :param sequences: list
        :return: list
        """
        if not sequences:
            return None
        keys = []
        for args in zip(*sequences):
            key = int(self.get_next_key())
            self.pc.submit(key, parallel_execute_wrapper, func, args)
            keys.append(key)
        return self.AsyncResultWrapper(self, keys)

    def get(self, object_name):
        """
        ParallelContext lacks a native method to get the value of an object from all workers. This method implements a
        synchronous (blocking) pull operation.
        :param object_name: str
        :return: dynamic
        """
        return self.apply_sync(find_nested_object, object_name)

    def synchronize(self, func, *args, **kwargs):
        """
        ParallelContext contains a native method to execute a function simultaneously on all workers except the master
        (root) rank. This method utilizes this method (pc.context) to execute a function with provided positional args
        and named kwargs. The executed function can include MPI operations that use the global communicator
        (interface.global_comm) to exchange data between all ranks across all ParallelContext subworlds. Unfortunately,
        this method cannot be used to collect return values from the executed function.
        :param func:
        :param args:
        :param kwargs:
        """
        self.pc.context(pc_synchronize_wrapper, func, args, kwargs)
        for _ in range(self.pc.nhost_bbs() - 1):
            self.pc.take("pc_synchronize")
        parallel_execute_wrapper(func, args, kwargs)

    def update_worker_contexts(self, content=None, **kwargs):
        """
        Data provided either through the positional argument content as a dictionary, or through kwargs, will be used to
        update the remote Context objects found on all ranks across all subworlds. Uses a global MPI broadcast
        operation.
        :param content: dict
        """
        if content is None:
            content = dict()
        content.update(kwargs)
        self.pc.context(pc_update_worker_contexts_wrapper)
        for _ in range(self.pc.nhost_bbs() - 1):
            self.pc.take("pc_update_worker_contexts")
        pc_update_worker_contexts_wrapper(content)

    def start(self, disp=False):
        if disp:
            self.print_info()
        self._running = True
        try:
            self.pc.runworker()
        except:
            self.hard_stop()

    def stop(self):
        self.pc.done()
        self._running = False
        self.h.quit()
        os._exit(1)

    def hard_stop(self):
        """
        Exceptions in python on an MPI rank are not enough to end a job. Strange behavior results when an unhandled
        Exception occurs on an MPI rank while running a neuron.h.ParallelContext.runworker() loop. This method will
        hard exit python.
        """
        print('nested: ParallelContextInterface: pid: %i; global_rank: %i brought down the whole operation' %
              (os.getpid(), self.global_rank))
        sys.stdout.flush()
        time.sleep(1.)
        os._exit(1)

    def ensure_controller(self):
        """
        Exceptions in python on an MPI rank are not enough to end a job. Strange behavior results when an unhandled
        Exception occurs on an MPI rank while running a neuron.h.ParallelContext.runworker() loop. This method will
        hard exit python if executed by any rank other than the master.
        """
        if self.global_rank != 0:
            self.hard_stop()


def pc_synchronize_wrapper(func, args, kwargs=None):
    """

    :param func: callable
    :param args: list
    :param kwargs: dict
    """
    interface = pc_find_interface()
    if interface.pc.id_bbs() > 0:
        interface.pc.post("pc_synchronize")
    discard = parallel_execute_wrapper(func, args, kwargs)


def parallel_execute_wrapper(func, args, kwargs=None):
    """
    When executing functions remotely, raised Exceptions do not necessarily result in an informative traceback. This
    wrapper is used by ParallelContextInterface and MPIFuturesInterface to first print a traceback on failed workers
    before the entire interface shuts down.
    :param func: callable
    :param args: list
    :param kwargs: dict
    :return: dynamic
    """
    if kwargs is None:
        kwargs = dict()
    try:
        result = func(*args, **kwargs)
    except Exception as e:
        print('nested: Exception occurred on process: %i. Waiting for pending jobs to complete' % os.getpid())
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        time.sleep(1.)
        raise e
    return result


def pc_apply_wrapper(func, key, args, kwargs):
    """
    Method used by ParallelContextInterface to implement an 'apply' operation. As long as a module executes 
    'from nested.parallel import *', this method can be executed remotely, and prevents any worker from returning until 
    all workers have applied the specified function.
    :param func: callable
    :param key: int
    :param args: list
    :param kwargs: dict
    :return: dynamic
    """
    interface = pc_find_interface()
    if interface.global_comm.rank == 0:
        interface.pc.master_works_on_jobs(0)
    if interface.pc.id_bbs() > 0:
        interface.pc.post(key)
    if interface.global_comm.rank == 0:
        for _ in range(interface.pc.nhost_bbs() - 1):
            interface.pc.take(key)
    result = parallel_execute_wrapper(func, args, kwargs)
    if interface.global_comm.rank == 0:
        interface.pc.master_works_on_jobs(1)
    sys.stdout.flush()
    return result


def pc_find_interface():
    """
    ParallelContextInterface apply and get operations require a remote instance of ParallelContextInterface. This method
    attempts to find it in the remote __main__ namespace, or in a Context object therein.
    :return: :class:'ParallelContextInterface'
    """
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
        return interface
    except Exception:
        raise Exception('nested: ParallelContextInterface: remote instance of ParallelContextInterface not found in '
                        'the remote __main__ namespace')


def pc_update_worker_contexts_wrapper(content=None):
    """
    nested.parallel interfaces require a remote instance of Context. This method can be used to update each remote
    Context with the contents of the provided dictionary.
    :param content: dict
    """
    interface = pc_find_interface()
    if interface.pc.id_bbs() > 0:
        interface.pc.post("pc_update_worker_contexts")
    content = interface.global_comm.bcast(content, root=0)
    update_worker_contexts(content)


class SerialInterface(object):
    """
    Class provides a serial interface to locally test parallelized code on a single process.
    """

    class AsyncResultWrapper(object):
        """
        When ready(), get() returns results as a list in the same order as submission.
        """

        def __init__(self, result):
            """

            :param result: iterator
            """
            self.result = list(result)

        def ready(self, **kwargs):
            """
            Serial operations are blocking, so results are always ready.
            :return: bool
            """
            return True

        def get(self):
            """
            Returns a list of results in the order of original submission.
            :return: list
            """
            return self.result

    def __init__(self):
        """

        """
        self.procs_per_worker = 1
        self.worker_id = 0
        self.num_workers = 1
        self.global_size = 1
        self.map_sync = lambda func, *args: list(map(func, *args))
        self.map = self.map_sync
        self.map_async = lambda func, *args: self.AsyncResultWrapper(self.map_sync(func, *args))
        self.apply_sync = lambda func, *args, **kwargs: [func(*args, **kwargs)]
        self.apply = self.apply_sync
        self.execute = lambda func, *args, **kwargs: func(*args, **kwargs)
        self.controller_is_worker = True

    def print_info(self):
        print('nested: SerialInterface: process id: %i' % os.getpid())
        sys.stdout.flush()
        time.sleep(0.1)

    def get(self, object_name):
        """
        This method implements a synchronous (blocking) pull operation.
        :param object_name: str
        :return: dynamic
        """
        return [self.execute(find_nested_object, object_name)]

    def update_worker_contexts(self, content=None, **kwargs):
        """
        Data provided either through the positional argument content as a dictionary, or through kwargs, will be used to
        update the remote Context objects found on all workers, using an apply operation.
        :param content: dict
        """
        if content is None:
            content = dict()
        content.update(kwargs)
        update_worker_contexts(content)

    def start(self, disp=False):
        if disp:
            self.print_info()

    def stop(self):
        os._exit(1)

    def ensure_controller(self):
        pass


def get_parallel_interface(framework='pc', procs_per_worker=1, source_file=None, source_package=None, sleep=0,
                           profile='default', cluster_id=None, **kwargs):
    """
    For convenience, scripts can be built with a click command line interface, and unknown command line arguments can
    be passed onto the appropriate constructor and return an instance of a ParallelInterface class.
    :param framework: str
    :param procs_per_worker: int
    :param source_file: str
    :param source_package: str
    :param sleep: int
    :param profile: str
    :param cluster_id: str
    :return: :class: 'IpypInterface', 'MPIFuturesInterface', 'ParallelContextInterface', or 'SerialInterface'
    """
    if framework == 'pc':
        return ParallelContextInterface(procs_per_worker=int(procs_per_worker))
    elif framework == 'mpi':
        return MPIFuturesInterface(procs_per_worker=int(procs_per_worker))
    elif framework == 'ipyp':
        return IpypInterface(cluster_id=cluster_id, profile=profile, procs_per_worker=int(procs_per_worker),
                             sleep=int(sleep), source_file=source_file, source_package=source_package)
    elif framework == 'serial':
        return SerialInterface()
    else:
        raise NotImplementedError('nested.parallel: interface for %s framework not yet implemented' % framework)
