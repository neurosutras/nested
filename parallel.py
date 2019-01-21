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

        def __init__(self, async_result):
            """
            :param async_result: :class:'ASyncResult'
            """
            self.async_result = async_result
            self._ready = False
            self.stdout = []

        def ready(self, wait=None):
            """

            :param wait: int or float
            :return: bool
            """
            if not self._ready:
                self.stdout = self.async_result.stdout
            else:
                return True
            self._ready = self.async_result.ready()
            if not self._ready and wait is not None:
                time.sleep(wait)
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
            """
            for stdout in self.stdout:
                if stdout:
                    for line in stdout.splitlines():
                        print line
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
            print 'nested: IpypInterface: procs_per_worker reduced to 1; collective operations not yet implemented'
        self.procs_per_worker = 1
        self.num_workers = self.global_size / self.procs_per_worker
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
        # print 'This is the file: %s; the dir: %s; the package: %s; the source: %s' % \
        #       (source_file, source_dir, source_package, source)
        try:
            self.direct_view[:].execute('from %s import *' % source, block=True)
            time.sleep(sleep)
        except Exception:
            raise Exception('nested.parallel: IPypInterface: failed to import source: %s from dir: %s' %
                            (source, source_dir))
        self.apply_sync = \
            lambda func, *args, **kwargs: \
                self._sync_wrapper(self.AsyncResultWrapper(self.direct_view[:].apply_async(func, *args, **kwargs)))
        self.apply = self.apply_sync
        self.map_sync = \
            lambda func, *args: self._sync_wrapper(self.AsyncResultWrapper(self.direct_view[:].map_async(func, *args)))
        self.map_async = lambda func, *args: self.AsyncResultWrapper(self.load_balanced_view.map_async(func, *args))
        self.map = self.map_sync
        self.get = lambda x: self.direct_view[:][x]
        self.print_info()

    def _sync_wrapper(self, async_result_wrapper):
        """

        :param async_result_wrapper: :class:'ASyncResultWrapper'
        :return: list
        """
        while not async_result_wrapper.ready():
            time.sleep(0.3)
        return async_result_wrapper.get()
    
    def print_info(self):
        print 'nested: IpypInterface: process id: %i; num workers: %i' % (os.getpid(), self.num_workers)
        sys.stdout.flush()

    def start(self, disp=False):
        pass

    def stop(self):
        os._exit(1)

    def ensure_controller(self):
        pass


class MPIFuturesInterface(object):
    """
    Class provides an interface to extend the mpi4py.futures concurrency tools for flexible nested parallel
    computations.
    """
    class AsyncResultWrapper(object):
        """
        When ready(), get() returns results as a list in the same order as submission.
        """

        def __init__(self, futures):
            """

            :param futures: list of :class:'mpi4py.futures.Future'
            """
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
            while not np.all([future.done() for future in self.futures]):
                if time.time() - time_stamp > wait:
                    return False
            self._ready = True
            return True

        def get(self):
            """
            Returns None until all results have completed, then returns a list of results in the order of original
            submission.
            :return: list
            """
            if self._ready or self.ready():
                return [future.result() for future in self.futures]
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
            print 'nested: MPIFuturesInterface: procs_per_worker reduced to 1; collective operations not yet ' \
                  'implemented'
        self.procs_per_worker = 1
        self.executor = MPIPoolExecutor()
        self.rank = self.global_comm.rank
        self.global_size = self.global_comm.size
        self.num_workers = self.global_size - 1
        self.apply_counter = 0
        self.map = self.map_sync
        self.apply = self.apply_sync
        self.init_workers(disp=True)

    def init_workers(self, disp=False):
        """

        :param disp: bool
        """
        futures = []
        for task_id in xrange(1, self.global_size):
            futures.append(self.executor.submit(mpi_futures_init_worker, task_id, disp))
        results = [future.result() for future in futures]
        num_returned = len(set(results))
        if num_returned != self.num_workers:
            raise ValueError('nested: MPIFuturesInterface: %i / %i processes returned from init_workers' %
                             (num_returned, self.num_workers))
        self.print_info()

    def print_info(self):
        """

        """
        print 'nested: MPIFuturesInterface: process id: %i; rank: %i / %i; num_workers: %i' % \
              (os.getpid(), self.rank, self.global_size, self.num_workers)
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
        for rank in xrange(1, self.global_size):
            futures.append(self.executor.submit(mpi_futures_apply_wrapper, func, apply_key, args, kwargs))
        results = [future.result() for future in futures]
        return results

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
        results = []
        for result in self.executor.map(func, *sequences):
            results.append(result)
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
            futures.append(self.executor.submit(func, *args))
        return self.AsyncResultWrapper(futures)

    def get(self, object_name):
        """
        mpi4py.futures lacks a native method to get the value of an object from all workers. This method implements a
        synchronous (blocking) pull operation.
        :param object_name: str
        :return: dynamic
        """
        return self.apply_sync(find_nested_object, object_name)

    def start(self, disp=False):
        pass

    def stop(self):
        self.executor.shutdown()
        # os._exit(0)

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
        open_ranks = range(2, comm.size)
        for worker_rank in open_ranks:
            future = comm.irecv(source=worker_rank)
            val = future.wait()
            if val != worker_rank:
                raise ValueError('nested: MPIFuturesInterface: process id: %i; rank: %i; received wrong value: %i; '
                                 'from worker: %i' % (os.getpid(), comm.rank, val, worker_rank))
        for worker_rank in open_ranks:
            comm.isend(key, dest=worker_rank)
        if disp:
            print 'Rank: %i took %.3f s to complete wait_for_all_workers' % (comm.rank, time.time() - start_time)
            sys.stdout.flush()
            time.sleep(0.1)
    else:
        comm.isend(comm.rank, dest=1)
        future = comm.irecv(source=1)
        val = future.wait()
        if val != key:
            raise ValueError('nested: MPIFuturesInterface: process id: %i; rank: %i; expected apply_key: '
                             '%i; received: %i from rank: 1' % (os.getpid(), comm.rank, key, val))


def mpi_futures_init_worker(task_id, disp=False):
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
        local_context.comm = MPI.COMM_SELF
    if task_id != local_context.global_comm.rank:
        raise ValueError('nested: MPIFuturesInterface: init_worker: process id: %i; rank: %i; received wrong task_id: '
                         '%i' % (os.getpid(), local_context.global_comm.rank, task_id))
    if disp:
        print 'nested: MPIFuturesInterface: process id: %i; rank: %i / %i; procs_per_worker: %i' % \
              (os.getpid(), local_context.global_comm.rank, local_context.global_comm.size, local_context.comm.size)
        sys.stdout.flush()
        time.sleep(0.1)
    return local_context.global_comm.rank


def find_context():
    """
    nested.parallel interfaces require a remote instance of Context. This method attempts to find it in the remote
    __main__ namespace.
    :return: :class:'Context'
    """
    try:
        module = sys.modules['__main__']
        local_context = None
        for item_name in dir(module):
            if isinstance(getattr(module, item_name), Context):
                local_context = getattr(module, item_name)
                break
    except Exception:
        raise Exception('nested.parallel: remote instance of Context not found in the remote __main__ namespace')
    return local_context


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
    result = func(*args, **kwargs)
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
            while len(self.remaining_keys) > 0 and self.interface.pc.working():
                key = int(self.interface.pc.userid())
                self.interface.collected[key] = self.interface.pc.pyret()
                try:
                    self.remaining_keys.remove(key)
                except ValueError:
                    pass
                if time.time() - time_stamp > wait:
                    return False
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
                except KeyError:
                    raise KeyError('nested: ParallelContextInterface: AsyncResultWrapper: all jobs have completed, but '
                                   'not all requested keys were found')
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
            raise ImportError('nested: ParallelContextInterface: problem with importing neuron')
        self.global_comm = MPI.COMM_WORLD
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
               self.global_comm.size / self.procs_per_worker == self.num_workers, \
            'nested: ParallelContextInterface: pc.ids do not match MPI ranks'
        self._running = False
        self.map = self.map_sync
        self.apply = self.apply_sync
        self.key_counter = 0
        self.maxint = 1e7

    def print_info(self):
        print 'nested: ParallelContextInterface: process id: %i; global rank: %i / %i; local rank: %i / %i; ' \
              'worker id: %i / %i' % \
              (os.getpid(), self.global_rank, self.global_size, self.comm.rank, self.comm.size, self.worker_id,
               self.num_workers)
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

    def wait_for_all_workers(self, key):
        """
        Prevents any worker from returning until all workers have completed the operation associated with the specified
        key.
        :param key: int or str
        """
        iter_count = 0
        self.pc.master_works_on_jobs(0)
        if self.rank == 0:
            self.pc.take(key)
            count = self.pc.upkscalar()
            count += 1
            self.pc.post(key, count)
            while count < self.num_workers:
                if self.pc.look(key):
                    count = self.pc.upkscalar()
                time.sleep(0.1)
                iter_count += 1
        self.pc.master_works_on_jobs(1)
        return

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
            apply_key = str(self.get_next_key())
            self.pc.post(apply_key, 0)
            keys = []
            for i in xrange(self.num_workers):
                key = int(self.get_next_key())
                self.pc.submit(key, pc_apply_wrapper, func, apply_key, args, kwargs)
                keys.append(key)
            results = self.collect_results(keys)
            self.pc.take(apply_key)
            sys.stdout.flush()
            return results
        else:
            result = func(*args, **kwargs)
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
        :return: dict
        """
        if keys is None:
            while self.pc.working():
                key = int(self.pc.userid())
                self.collected[key] = self.pc.pyret()
            keys = self.collected.keys()
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
            try:
                return [self.collected.pop(key) for key in keys]
            except KeyError:
                raise KeyError('nested: ParallelContextInterface: all jobs have completed, but not all requested keys '
                               'were found')

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
            self.pc.submit(key, func, *args)
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
            self.pc.submit(key, func, *args)
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
        hard exit python if executed by any rank other than the master.
        """
        print 'nested: ParallelContextInterface: pid: %i; global_rank: %i brought down the whole operation' % \
              (os.getpid(), self.global_rank)
        os._exit(1)

    def ensure_controller(self):
        """
        Exceptions in python on an MPI rank are not enough to end a job. Strange behavior results when an unhandled
        Exception occurs on an MPI rank while running a neuron.h.ParallelContext.runworker() loop. This method will
        hard exit python if executed by any rank other than the master.
        """
        if self.global_rank != 0:
            print 'nested: ParallelContextInterface: pid: %i; global_rank: %i brought down the whole operation' % \
                  (os.getpid(), self.global_rank)
            os._exit(1)


def pc_apply_wrapper(func, key, args, kwargs):
    """
    Method used by ParallelContextInterface to implement an 'apply' operation. As long as a module executes 
    'from nested.parallel import *', this method can be executed remotely, and prevents any worker from returning until 
    all workers have applied the specified function.
    :param func: callable
    :param key: int or str
    :param args: list
    :param kwargs: dict
    :return: dynamic
    """
    interface = pc_find_interface()
    interface.wait_for_all_workers(key)
    result = func(*args, **kwargs)
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
