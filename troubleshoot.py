from parallel import *
import click


context = Context()


def collect_ranks(tag):
    """

    :param tag: int
    :return: tuple: int, dict
    """
    time.sleep(0.1)
    start_time = time.time()
    ranks = context.interface.comm.gather(context.interface.global_rank, root=0)
    if context.interface.rank == 0:
        return context.interface.worker_id, {'ranks': ranks, 'tag': int(tag), 'compute time': time.time() - start_time}


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
          (context.interface.global_rank, context.interface.global_size, context.interface.rank, context.interface.size,
           context.interface.worker_id, context.interface.num_workers, context.count)


@click.command()
@click.option("--procs-per-worker", type=int, default=1)
def main(procs_per_worker):
    """

    :param procs_per_worker: int
    """
    
    try:
        from mpi4py import MPI
        from neuron import h
    except ImportError:
        raise ImportError('nested: ParallelContextInterface: problem with importing neuron')
    global_comm = MPI.COMM_WORLD
    pc = h.ParallelContext()
    # pc.subworlds(procs_per_worker)   
    global_rank = int(pc.id_world())
    global_size = int(pc.nhost_world())
    rank = int(pc.id())
    size = int(pc.nhost())
    for i in xrange(global_comm.size):
        if global_comm.rank == i:
            print global_comm.rank, global_comm.size, rank, size, global_rank, global_size
            time.sleep(1.)
    
    """    
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
        self.apply_counter = 0
    
    context.interface = ParallelContextInterface(procs_per_worker=procs_per_worker)
    results1 = context.interface.apply(set_count)
    if context.interface.global_rank == 0:
        print 'before interface.start(): context.interface.apply(set_count)'
        pprint.pprint(results1)
    time.sleep(0.1)
    context.interface.start()
    results2 = context.interface.apply(set_count, 5)
    print 'after interface.start(): context.interface.apply(set_count, 5)'
    pprint.pprint(results2)
    results3 = context.interface.map_sync(collect_ranks, range(10))
    print ': context.interface.map_sync(collect_ranks, range(10))'
    pprint.pprint(results3)
    results4 = context.interface.map_async(collect_ranks, range(10, 20))
    print 'collected result keys: %s' % str(context.interface.collected.keys())
    while not results4.ready():
        pass
    results4 = results4.get()
    print ': context.interface.map_async(collect_ranks, range(10, 20))'
    pprint.pprint(results4)
    print 'collected result keys: %s' % str(context.interface.collected.keys())
    results5 = context.interface.apply(collect_ranks, 0)
    print ': context.interface.apply(collect_ranks, 0)'
    pprint.pprint(results5)
    context.interface.stop()
    """

if __name__ == '__main__':
    main()
