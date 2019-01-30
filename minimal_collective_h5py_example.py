"""
Requires mpi4py, parallel hdf5, parallel h5py, and NEURON.
Uses the nested.parallel.ParallelContext interface.
Example script demonstrates a workaround to do collective write operations when using the NEURON ParallelContext
bulletin board.
"""
__author__ = 'Aaron D. Milstein and Grace Ng'
from nested.optimize_utils import *
from nested.parallel import ParallelContextInterface
import click
import uuid


context = Context()


@click.command()
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--verbose", type=int, default=1)
def main(output_dir, verbose):
    """

    :param output_dir: str (path)
    :param verbose: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    disp = verbose > 0

    context.interface = ParallelContextInterface()
    context.interface.apply(config_parallel_interface, __file__, output_dir=output_dir, disp=disp, verbose=verbose)
    context.interface.start(disp=True)
    context.interface.ensure_controller()

    for i in xrange(2):
        context.interface.apply(test_collective_write)
        context.interface.map(read_open_file, range(200))

    context.interface.apply(shutdown_worker)
    context.interface.stop()


def config_worker():

    context.temp_model_data = dict()
    start = context.interface.global_comm.rank
    id = uuid.uuid1()
    context.temp_model_data[id] = np.linspace(start, start + 1, 100)
    context.temp_model_data_file_path = None


def shutdown_worker():
    context.temp_model_data_file.close()
    if context.interface.global_comm.rank == 0:
        os.remove(context.temp_model_data_file_path)


def read_open_file(i):

    this_data = dict()
    for model_key in context.temp_model_data:
        group_key = context.temp_model_data_legend[model_key]
        this_data[model_key] = context.temp_model_data_file[group_key][:]
    if context.verbose > 1:
        print 'Rank: %i; data: %s' % (context.interface.global_comm.rank, str(this_data))


def test_collective_write():
    """

    """
    context.temp_model_data_legend = dict()
    if context.interface.global_comm.rank == 0:
        context.interface.pc.post('test1', context.temp_model_data)
    elif context.interface.global_comm.rank == 1:
        context.interface.pc.take('test1')
        temp_model_data_from_master = context.interface.pc.upkpyobj()
        context.temp_model_data.update(temp_model_data_from_master)
    if context.interface.global_comm.rank > 0:
        temp_model_data_keys = context.temp_model_data.keys()
        temp_model_data_keys = context.interface.worker_comm.gather(temp_model_data_keys, root=0)
        if context.interface.worker_comm.rank == 0:
            key_list = list(set([key for key_list in temp_model_data_keys for key in key_list]))
        else:
            key_list = None
        key_list = context.interface.worker_comm.bcast(key_list, root=0)
        if context.temp_model_data_file_path is None:
            if context.interface.worker_comm.rank == 0:
                context.temp_model_data_file_path = '%s/%s_temp_model_data.hdf5' % \
                                               (context.output_dir, datetime.datetime.today().strftime('%Y%m%d_%H%M'))
            context.temp_model_data_file_path = \
                context.interface.worker_comm.bcast(context.temp_model_data_file_path, root=0)
            context.temp_model_data_file = h5py.File(context.temp_model_data_file_path, 'a', driver='mpio',
                                                     comm=context.interface.worker_comm)
        for i, model_key in enumerate(key_list):
            group_key = str(i)
            context.temp_model_data_legend[model_key] = group_key
            if group_key not in context.temp_model_data_file:
                context.temp_model_data_file.create_dataset(group_key, (100,))
            if model_key in context.temp_model_data:
                context.temp_model_data_file[group_key][:] = context.temp_model_data[model_key]
        context.temp_model_data_file.flush()

    if context.interface.global_comm.rank == 0:
        context.interface.pc.take('test2')
        context.temp_model_data_legend = context.interface.pc.upkpyobj()
        context.interface.pc.take('test3')
        context.temp_model_data_file_path = context.interface.pc.upkpyobj()[0]
        if 'temp_model_data_file' not in context() or context.temp_model_data_file is None:
            context.temp_model_data_file = h5py.File(context.temp_model_data_file_path, 'r')
    elif context.interface.global_comm.rank == 1:
        context.interface.pc.post('test2', context.temp_model_data_legend)
        context.interface.pc.post('test3', [context.temp_model_data_file_path])


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
