from nested.utils import *


context = Context()


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
        self.apply_counter = 1
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
        mpi_futures_wait_for_all_workers(self.global_comm, apply_key, disp=True)
        results = [future.result() for future in futures]
        print 'Rank: %i got all the results back from apply' % self.global_comm.rank
        sys.stdout.flush()
        time.sleep(0.1)
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

    :param comm: :class:'MPI.COMM_WORLD'
    :param key: int
    :param disp: bool; verbose reporting for debugging
    """
    start_time = time.time()
    if disp:
        print 'Rank: %i entered wait_for_all_workers loop' % (comm.rank)
        sys.stdout.flush()
    if comm.rank > 0:
        tag = key * comm.rank
        comm.isend(key, dest=0, tag=tag)
        while not comm.iprobe(source=0, tag=tag):
            time.sleep(0.1)
        req = comm.irecv(source=0, tag=tag)
        recv_key = req.wait()
        if recv_key != key:
            raise ValueError('nested: MPIFuturesInterface: process id: %i; rank: %i; expected apply_key: %i; received: '
                             '%i from rank: %i' % (os.getpid(), comm.rank, key, recv_key, 0))
    else:
        remaining = range(1, comm.size)
        while len(remaining) > 0:
            for rank in remaining:
                tag = key * rank
                if comm.iprobe(source=rank, tag=tag):
                    req = comm.irecv(source=rank, tag=tag)
                    recv_key = req.wait()
                    if recv_key != key:
                        raise ValueError('nested: MPIFuturesInterface: process id: %i; rank: %i; expected apply_key: '
                                         '%i; received: %i from rank: %i' %
                                         (os.getpid(), comm.rank, key, recv_key, rank))
                    remaining.remove(rank)
        time.sleep(0.1)
        if disp:
            print 'nested: MPIFuturesInterface: process id: %i; rank: %i; received all messages' % \
                  (os.getpid(), comm.rank)
            sys.stdout.flush()
            time.sleep(0.1)
        for rank in range(1, comm.size):
            tag = key * rank
            comm.isend(key, dest=rank, tag=tag)
        if disp:
            print 'nested: MPIFuturesInterface: process id: %i; rank: %i; sent all messages' % \
                  (os.getpid(), comm.rank)
            sys.stdout.flush()
            time.sleep(0.1)
    if disp:
        print 'Rank: %i took %.2f s to complete wait_for_all_workers loop' % \
              (comm.rank, time.time() - start_time)
        sys.stdout.flush()
        time.sleep(0.1)


def mpi_futures_init_worker(task_id, disp=False):
    """
    Create an MPI communicator and insert it into a local Context object on each remote worker.
    :param task_id: int
    :param disp: bool
    """
    local_context = mpi_futures_find_context()
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


def mpi_futures_find_context():
    """
    MPIFuturesInterface apply and get operations require a remote instance of Context. This method attempts to find it
    in the remote __main__ namespace.
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
        raise Exception('nested: MPIFuturesInterface: remote instance of Context not found in the remote __main__ '
                        'namespace')
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
    local_context = mpi_futures_find_context()
    mpi_futures_wait_for_all_workers(local_context.global_comm, key, disp=True)
    print 'Rank: %i returned from wait_for_all_workers' % local_context.global_comm.rank
    sys.stdout.flush()
    time.sleep(0.1)
    result = func(*args, **kwargs)
    print 'Rank: %i executed the function: %s' % (local_context.global_comm.rank, str(func))
    sys.stdout.flush()
    time.sleep(0.1)
    # return func(*args, **kwargs)
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


def report_rank():
    return context.global_comm.rank


def main():
    context.interface = MPIFuturesInterface()
    print ':context.interface.apply(report_rank)'
    results = context.interface.apply(report_rank)
    num_returned = len(set(results))
    print 'nested: MPIFuturesInterface: %i / %i workers participated in apply' % \
          (num_returned, context.interface.num_workers)
    sys.stdout.flush()
    time.sleep(1.)
    context.interface.stop()


if __name__ == '__main__':
    main()