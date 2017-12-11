#ipcontroller --nodb --ip='*'&
#mpirun -n 3 ipengine --mpi='mpi4py'

import importlib
from moopgen import *
from ipyparallel import Client

kwargs = {'cvode': False, 'verbose': False, 'daspk': True}

global_context = Context()
global_context.kwargs = kwargs
global_context.sleep = False

def init_controller():
    module_set = set(['simple_network_submodule'])
    global_context.module_set = module_set
    for module_name in module_set:
        m = importlib.import_module(module_name)
        m.config_controller(**global_context.kwargs)
        print 'imported module'
    get_features_func = getattr(m, 'get_feature')
    global_context.get_features_func = get_features_func
    print 'initiated controller'
    sys.stdout.flush()

def setup_client_interface():
    c = Client()
    print 'num procs %d' %len(c)
    sys.stdout.flush()
    global_context.c = c
    global_context.c[:].execute('from simple_test_optimize_network import *', block=True)
    print 'Got past module import'
    if global_context.sleep:
        time.sleep(120.)
    global_context.c[:].apply_sync(init_engine, global_context.module_set, **global_context.kwargs)
    result = global_context.c[:].map_async(sys.modules['simple_network_submodule'].report_pc_id, range(len(c)))
    while not result.ready():
        for stdout in result.stdout:
            if stdout:
                for line in stdout.splitlines():
                    print line
        sys.stdout.flush()
        time.sleep(1.)
    print result.get()

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
    setup_client_interface()
    run_optimization()
