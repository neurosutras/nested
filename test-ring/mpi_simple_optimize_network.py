#mpiexec -n 1 -usize=5 python mpi_simple_optimize_network.py

from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
import importlib
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimize_utils import *


kwargs = {'cvode': False, 'verbose': False, 'daspk': True}

global_context = Context()
global_context.kwargs = kwargs
global_context.sleep = False

comm = MPI.COMM_WORLD
rank = comm.rank

def init_controller():
    module_set = set(['mpi_simple_network_submodule'])
    # module_set = set(['dummy_submodule'])
    global_context.module_set = module_set
    for module_name in module_set:
        m = importlib.import_module(module_name)
        m.config_controller(**global_context.kwargs)
        print 'imported module'
    # get_features_func = getattr(m, 'get_feature')
    # global_context.get_features_func = get_features_func
    print 'initiated controller'
    sys.stdout.flush()
"""
def setup_client_interface():
    c = MPIPoolExecutor()
    global_context.c = c
    results = c.map(report_id)
    for result in results:
        print result
    sys.stdout.flush()
    #c.submit(init_engine, global_context.module_set, **global_context.kwargs)
    #result = c.submit(sys.modules['mpi_simple_network_submodule'].report_pc_id)
"""

def report_id(id):
    #comm = MPI.COMM_WORLD
    #rank = comm.rank
    #return type(rank)
    pid = os.getpid()
    return {'rank': rank, 'id': id, 'size': comm.size, 'pid': pid}


def init_engine(module_set, **kwargs):
    print 'called init_engine'
    for module_name in module_set:
        m = importlib.import_module(module_name)
        config_func = getattr(m, 'config_engine')
        if not callable(config_func):
            raise Exception('parallel_optimize: init_engine: submodule: %s does not contain required callable: '
                            'config_engine' % module_name)
        else:
            config_func(**kwargs)

def run_optimization():
    generation = [0, 1, 2]
    features = get_all_features(generation)
    print features

def get_all_features(generation):
    pop_ids = range(len(generation))
    curr_generation = {pop_id: generation[pop_id] for pop_id in pop_ids}
    results = []
    features_dict = {pop_id: {} for pop_id in pop_ids}
    indivs = [{'pop_id': pop_id, 'x': curr_generation[pop_id],
               'features': features_dict[pop_id]} for pop_id in curr_generation]
    feature_function = global_context.get_features_func
    #results.extend(map(feature_function, indivs, [global_context.c]*len(pop_ids), [[0], [1], [2]]))
    feature_function(indivs, global_context.c)
    results2 = []
    feature_function = global_context.get_features_func
    results2.extend(map(feature_function(indivs)))
    return [results, results2]

if __name__ == '__main__':
    init_controller()
    print rank, comm.size
    sys.stdout.flush()
    c1 = MPIPoolExecutor(max_workers=2)
    c2 = MPIPoolExecutor(max_workers=2)
    #global_context.c = c
    # c1.map(init_engine(global_context.module_set))
    results = c1.map(report_id, range(2))
    for result in results:
        print result
    results = c2.map(report_id, range(2, 4))
    for result in results:
        print result
    sys.stdout.flush()
    #setup_client_interface()
    #run_optimization()
