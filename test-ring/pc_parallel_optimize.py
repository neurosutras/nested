#mpiexec -n 4 python pc_parallel_optimize.py --config-file-path simple_test_config.yaml --pop-size 3 --max-iter 3

from mpi4py import MPI
from neuron import h
import click
import importlib
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimize_utils import *

script_filename = 'pc_parallel_optimize.py'
global_context = Context()
comm = MPI.COMM_WORLD
global_context.comm = comm


@click.command()
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None)
@click.option("--param-gen", type=str, default='BGen')
@click.option("--pop-size", type=int, default=100)
@click.option("--wrap-bounds", is_flag=True)
@click.option("--seed", type=int, default=None)
@click.option("--max-iter", type=int, default=None)
@click.option("--path-length", type=int, default=1)
@click.option("--initial-step-size", type=float, default=0.5)
@click.option("--adaptive-step-factor", type=float, default=0.9)
@click.option("--m0", type=int, default=20)
@click.option("--c0", type=int, default=20)
@click.option("--p_m", type=float, default=0.5)
@click.option("--delta_m", type=int, default=0)
@click.option("--delta_c", type=int, default=0)
@click.option("--mutate_survivors", is_flag=True)
@click.option("--survival-rate", type=float, default=0.2)
@click.option("--sleep", is_flag=True)
@click.option("--analyze", is_flag=True)
@click.option("--hot-start", is_flag=True)
@click.option("--storage-file-path", type=str, default=None)
@click.option("--export", is_flag=True)
@click.option("--output-dir", type=str, default='data')
@click.option("--export-file-path", type=str, default=None)
@click.option("--disp", is_flag=True)
def main(config_file_path, param_gen, pop_size, wrap_bounds, seed, max_iter, path_length, initial_step_size,
         adaptive_step_factor, m0, c0, p_m, delta_m, delta_c, mutate_survivors, survival_rate, sleep, analyze,
         hot_start, storage_file_path, export, output_dir, export_file_path, disp):
    """

    :param config_file_path: str (path)
    :param param_gen: str (must refer to callable in globals())
    :param pop_size: int
    :param wrap_bounds: bool
    :param seed: int
    :param max_iter: int
    :param path_length: int
    :param initial_step_size: float in [0., 1.]  # BGen-specific argument
    :param adaptive_step_factor: float in [0., 1.]  # BGen-specific argument
    :param m0: int : initial strength of mutation  # EGen-specific argument
    :param c0: int : initial strength of crossover  # EGen-specific argument
    :param p_m: float in [0., 1.] : probability of mutation  # EGen-specific argument
    :param delta_m: int : decrease mutation strength every interval  # EGen-specific argument
    :param delta_c: int : decrease crossover strength every interval  # EGen-specific argument
    :param mutate_survivors: bool  # EGen-specific argument
    :param survival_rate: float
    :param sleep: bool
    :param analyze: bool
    :param hot_start: bool
    :param storage_file_path: str
    :param export: bool
    :param output_dir: str
    :param export_file_path: str
    :param disp: bool
    """
    process_params(config_file_path, param_gen, pop_size, path_length, sleep, storage_file_path,
                   output_dir, export_file_path, disp)
    if analyze and export:
        global_context.pop_size = 1
        global_context.path_length = 1
        global_context.max_iter = 1
        setup_ranks()
    elif not analyze:
        setup_ranks()
    global storage
    global x_dict
    global x_array
    global best_indiv
    if not analyze:
        global param_gen_instance
        if hot_start:
            param_gen_instance = global_context.param_gen_class(
                pop_size=pop_size, x0=param_dict_to_array(global_context.x0, global_context.param_names),
                bounds=global_context.bounds, rel_bounds=global_context.rel_bounds, wrap_bounds=wrap_bounds, seed=seed,
                max_iter=max_iter, adaptive_step_factor=adaptive_step_factor, p_m=p_m, delta_m=delta_m, delta_c=delta_c,
                mutate_survivors=mutate_survivors, survival_rate=survival_rate, disp=disp, hot_start=storage_file_path,
                **global_context.kwargs)
        else:
            param_gen_instance = global_context.param_gen_class(
                param_names=global_context.param_names, feature_names=global_context.feature_names,
                objective_names=global_context.objective_names, pop_size=pop_size,
                x0=param_dict_to_array(global_context.x0, global_context.param_names), bounds=global_context.bounds,
                rel_bounds=global_context.rel_bounds, wrap_bounds=wrap_bounds, seed=seed, max_iter=max_iter,
                path_length=path_length, initial_step_size=initial_step_size, m0=m0, c0=c0, p_m=p_m, delta_m=delta_m,
                delta_c=delta_c, mutate_survivors=mutate_survivors, adaptive_step_factor=adaptive_step_factor,
                survival_rate=survival_rate, disp=disp, **global_context.kwargs)
        run_optimization()
        storage = param_gen_instance.storage
        best_indiv = storage.get_best(1, 'last')[0]
        x_array = best_indiv.x
        x_dict = param_array_to_dict(x_array, storage.param_names)
        if disp:
            print 'parallel_optimize: best params:'
            pprint.pprint(x_dict)
    elif storage_file_path is not None and os.path.isfile(storage_file_path):
        storage = PopulationStorage(file_path=storage_file_path)
        print 'parallel_optimize: analysis mode: history loaded from path: %s' % storage_file_path
        best_indiv = storage.get_best(1, 'last')[0]
        x_array = best_indiv.x
        x_dict = param_array_to_dict(x_array, storage.param_names)
        init_engine_interactive(x_dict)
        if disp:
            print 'parallel_optimize: best params:'
            pprint.pprint(x_dict)
    else:
        print 'parallel_optimize: analysis mode: no optimization history loaded'
        x_dict = global_context.x0
        x_array = param_dict_to_array(x_dict, global_context.param_names)
        init_engine_interactive(x_dict)
        if disp:
            print 'parallel_optimize: initial params:'
            pprint.pprint(x_dict)
    sys.stdout.flush()
    if export:
        global_context.exported_features, global_context.exported_objectives = export_traces(x_array)


def process_params(config_file_path, param_gen, pop_size, path_length, sleep, storage_file_path, output_dir,
                   export_file_path, disp):
    """

    :param config_file_path: str
    :param param_gen: str
    :param pop_size: int
    :param path_length: int
    :param sleep: bool
    :param storage_file_path: str
    :param output_dir: str
    :param export_file_path: str
    :param disp: bool
    """
    if config_file_path is None:
        raise Exception('parallel_optimize: a config_file_path specifying optimization parameters must be provided.')

    config_dict = read_from_yaml(config_file_path)
    if 'param_gen' in config_dict and config_dict['param_gen'] is not None:
        param_gen_name = config_dict['param_gen']
    else:
        param_gen_name = param_gen
    param_names = config_dict['param_names']
    if 'default_params' not in config_dict or config_dict['default_params'] is None:
        default_params = {}
    else:
        default_params = config_dict['default_params']
    for param in default_params:
        config_dict['bounds'][param] = (default_params[param], default_params[param])
    bounds = [config_dict['bounds'][key] for key in param_names]
    if 'rel_bounds' not in config_dict or config_dict['rel_bounds'] is None:
        rel_bounds = None
    else:
        rel_bounds = config_dict['rel_bounds']
    if 'x0' not in config_dict or config_dict['x0'] is None:
        x0 = None
    else:
        x0 = config_dict['x0']
    feature_names = config_dict['feature_names']
    objective_names = config_dict['objective_names']
    target_val = config_dict['target_val']
    target_range = config_dict['target_range']
    optimization_title = config_dict['optimization_title']
    kwargs = config_dict['kwargs']  # Extra arguments to be passed to imported submodules

    config_file_check = True
    missing_config = []
    """
    if 'update_params' not in config_dict or config_dict['update_params'] is None:
        update_params = []
    else:
        update_params = config_dict['update_params']
    if 'update_modules' not in config_dict or config_dict['update_modules'] is None:
        update_modules = []
    else:
        update_modules = config_dict['update_modules']
    """
    if 'get_features' not in config_dict or config_dict['get_features'] is None:
        config_file_check = False
        missing_config.append('get_features')
    else:
        get_features = config_dict['get_features']
    if 'features_modules' not in config_dict or config_dict['features_modules'] is None:
        config_file_check = False
        missing_config.append('features_modules')
    else:
        features_modules = config_dict['features_modules']
    if 'objectives_modules' not in config_dict or config_dict['objectives_modules'] is None:
        config_file_check = False
        missing_config.append('objectives_modules')
    else:
        objectives_modules = config_dict['objectives_modules']
    if 'subworld_size' not in config_dict or config_dict['subworld_size'] is None:
        config_file_check = False
        missing_config.append('subworld_size')
    else:
        subworld_size = config_dict['subworld_size']
    if not config_file_check:
        raise Exception('parallel_optimize: config_file at path: %s is missing the following required fields: %s' %
                        (config_file_path, ', '.join(str(field) for field in missing_config)))

    if storage_file_path is None:
        storage_file_path = '%s/%s_%s_optimization_history.hdf5' % \
                            (output_dir, optimization_title, param_gen_name)
    if export_file_path is None:
        export_file_path = '%s/%s_%s_%s_optimization_exported_output.hdf5' % \
                           (output_dir, datetime.datetime.today().strftime('%m%d%Y%H%M'), optimization_title,
                            param_gen_name)
    temp_output_path = '%s/parallel_optimize_temp_output.hdf5' % output_dir
    if param_gen_name not in globals():
        raise Exception('parallel_optimize: %s has not been imported, or is not a valid class of parameter '
                        'generator.' % param_gen_name)
    # param_gen_class points to the parameter generator class, while param_gen_name points to its name as a string
    param_gen_class = globals()[param_gen_name]

    global_context.update(locals())
    global_context.update(kwargs)
    print 'finished processing_params'
    sys.stdout.flush()


def setup_ranks():
    """
    if len(global_context.update_params) != len(global_context.update_modules):
        raise Exception('parallel_optimize: number of arguments in update_params does not match number of imported '
                        'submodules.')
    """
    if len(global_context.get_features) != len(global_context.features_modules):
        raise Exception('parallel_optimize: number of arguments in get_features does not match number of imported '
                        'submodules.')
    #module_set = set(global_context.update_modules)
    module_set = set([])
    module_set.update(global_context.features_modules, global_context.objectives_modules)
    global_context.module_set = module_set
    for module_name in module_set:
        m = importlib.import_module(module_name)
        """
        m.config_controller(global_context.export_file_path, output_dir=global_context.output_dir,
                            **global_context.kwargs)
        """
    """
    update_params_funcs = []
    for i, module_name in enumerate(global_context.update_modules):
        module = sys.modules[module_name]
        func = getattr(module, global_context.update_params[i])
        if not callable(func):
            raise Exception('parallel_optimize: update_params: %s for submodule %s is not a callable function.'
                            % (global_context.update_params[i], module_name))
        update_params_funcs.append(func)
    global_context.update_params_funcs = update_params_funcs
    """
    get_features_funcs = []
    for i, module_name in enumerate(global_context.features_modules):
        module = sys.modules[module_name]
        func = getattr(module, global_context.get_features[i])
        if not callable(func):
            raise Exception('parallel_optimize: get_features: %s for submodule %s is not a callable function.'
                            % (global_context.get_features[i], module_name))
        get_features_funcs.append(func)
    global_context.get_features_funcs = get_features_funcs
    get_objectives_funcs = []
    for module_name in global_context.objectives_modules:
        module = sys.modules[module_name]
        func = getattr(module, 'get_objectives')
        if not callable(func):
            raise Exception('parallel_optimize: submodule %s does not contain a required callable function '
                            'get_objectives.' % module_name)
        get_objectives_funcs.append(func)
    global_context.get_objectives_funcs = get_objectives_funcs

    if global_context.comm.rank == 0:
        print 'parallel_optimize: %s; parameter generator: %s; num_processes: %i; population size: %i; subworld size: %d;' \
              ' feature calculators: %s; imported submodules: %s' \
              % (global_context.optimization_title, global_context.param_gen_name, global_context.comm.size,
                 global_context.pop_size, global_context.subworld_size,
                 ', '.join(func_name for func_name in global_context.get_features),
                 ', '.join(name for name in global_context.module_set))
    init_engine(**global_context.kwargs)
    print 'finished setting up ranks'

def init_engine(**kwargs):
    for module_name in global_context.module_set:
        m = importlib.import_module(module_name)
        config_func = getattr(m, 'config_engine')
        if not callable(config_func):
            raise Exception('parallel_optimize: init_engine: submodule: %s does not contain required callable: '
                            'config_engine' % module_name)
        else:
            config_func(global_context.comm, global_context.subworld_size, global_context.target_val,
                        global_context.target_range, global_context.param_names, global_context.default_params,
                        global_context.temp_output_path, global_context.export_file_path, global_context.output_dir,
                        global_context.disp, **kwargs)
    sys.stdout.flush()


def run_optimization():
    print 'in run_optimization function'
    for ind, generation in enumerate(param_gen_instance()):
        if (ind > 0) and (ind % global_context.path_length == 0):
            param_gen_instance.storage.save(global_context.storage_file_path, n=global_context.path_length)
        features = get_all_features(generation)
        features, objectives = get_all_objectives(features)
        param_gen_instance.update_population(features, objectives)
    for module_name in global_context.module_set:
        m = sys.modules[module_name]
        getattr(m, 'end_optimization')()
    param_gen_instance.storage.save(global_context.storage_file_path, n=global_context.path_length)


def get_all_features(generation):
    """
    Note: differs from old parallel_optimize script in that we are no longer mapping each indiv to a separate feature_function call
    :param generation:
    :return:
    """
    disp = global_context.disp
    pop_ids = range(len(generation))
    curr_generation = {pop_id: generation[pop_id] for pop_id in pop_ids}
    features_dict = {pop_id: {} for pop_id in pop_ids}

    for ind in xrange(len(global_context.get_features_funcs)):
        next_generation = {}
        indivs = [{'pop_id': pop_id, 'x': curr_generation[pop_id], 'features': features_dict[pop_id]}
                  for pop_id in curr_generation]
        feature_function = global_context.get_features_funcs[ind]
        if disp:
            print 'start round %i' %ind
        results = feature_function(indivs)
        for i, result in enumerate(results):
            if None in result['result_list']:
                if disp:
                    print 'Individual: %i failed %s' %(result['pop_id'], global_context.get_features[ind])
                features_dict[result['pop_id']] = None
            else:
                next_generation[result['pop_id']] = generation[result['pop_id']]
                new_features = {key: value for result_dict in result['result_list'] for key, value in result_dict.iteritems()}
                features_dict[result['pop_id']].update(new_features)
        curr_generation = next_generation
    features = features_dict.values()
    print 'features'
    print features
    return features


def get_all_objectives(features):
    objectives_dict = {pop_id: {} for pop_id in range(len(features))}
    for objective_function in global_context.get_objectives_funcs:
        new_features, new_objectives = objective_function(features)
        for pop_id, objective in new_objectives.iteritems():
            objectives_dict[pop_id].update(objective)
            features[pop_id] = new_features[pop_id]
    objectives = objectives_dict.values()
    print 'objectives'
    print objectives
    return features, objectives


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_filename) != -1,sys.argv)+1):])