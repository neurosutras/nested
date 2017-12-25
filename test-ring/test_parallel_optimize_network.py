__author__ = 'Grace Ng and Aaron D. Milstein'
import click
from ipyparallel import Client
from moopgen import *
# from plot_results import *
import importlib

"""
Dynamically submits jobs to available cores.

Assumes a controller is already running in another process with:
ipcluster start -n num_cores
"""

try:
    import mkl

    mkl.set_num_threads(1)
except:
    pass

script_filename = 'parallel_optimize.py'

global_context = Context()


@click.command()
@click.option("--cluster-id", type=str, default=None)
@click.option("--profile", type=str, default='default')
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
def main(cluster_id, profile, config_file_path, param_gen, pop_size, wrap_bounds, seed, max_iter, path_length,
         initial_step_size, adaptive_step_factor, m0, c0, p_m, delta_m, delta_c, mutate_survivors, survival_rate, sleep,
         analyze, hot_start, storage_file_path, export, output_dir, export_file_path, disp):
    """

    :param cluster_id: str (optional, must match cluster-id of running ipcontroller or ipcluster)
    :param profile: str (optional, must match existing ipyparallel profile)
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
    process_params(cluster_id, profile, config_file_path, param_gen, pop_size, path_length, sleep, storage_file_path,
                   output_dir, export_file_path, disp)
    init_controller()
    if analyze and export:
        global_context.pop_size = 1
        global_context.path_length = 1
        global_context.max_iter = 1
        setup_client_interface()
    elif not analyze:
        setup_client_interface()
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


def process_params(cluster_id, profile, config_file_path, param_gen, pop_size, path_length, sleep, storage_file_path,
                   output_dir, export_file_path, disp):
    """

    :param cluster_id: str
    :param profile: str
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
    if 'update_params' not in config_dict or config_dict['update_params'] is None:
        update_params = []
    else:
        update_params = config_dict['update_params']
    if 'update_modules' not in config_dict or config_dict['update_modules'] is None:
        update_modules = []
    else:
        update_modules = config_dict['update_modules']
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
    if 'group_sizes' not in config_dict or config_dict['group_sizes'] is None:
        config_file_check = False
        missing_config.append('group_sizes')
    else:
        group_sizes = config_dict['group_sizes']
    if not config_file_check:
        raise Exception('parallel_optimize: config_file at path: %s is missing the following required fields: %s' %
                        (config_file_path, ', '.join(str(field) for field in missing_config)))

    if storage_file_path is None:
        storage_file_path = '%s/%s_%s_%s_optimization_history.hdf5' % \
                            (output_dir, datetime.datetime.today().strftime('%m%d%Y%H%M'), optimization_title,
                             param_gen_name)
    if export_file_path is None:
        export_file_path = '%s/%s_%s_%s_optimization_exported_output.hdf5' % \
                           (output_dir, datetime.datetime.today().strftime('%m%d%Y%H%M'), optimization_title,
                            param_gen_name)

    if param_gen_name not in globals():
        raise Exception('parallel_optimize: %s has not been imported, or is not a valid class of parameter '
                        'generator.' % param_gen_name)
    # param_gen_class points to the parameter generator class, while param_gen_name points to its name as a string
    param_gen_class = globals()[param_gen_name]

    global_context.update(locals())
    global_context.update(kwargs)
    sys.stdout.flush()


def init_controller():
    """

    """
    update_params = global_context.update_params
    update_modules = global_context.update_modules
    get_features = global_context.get_features
    features_modules = global_context.features_modules
    objectives_modules = global_context.objectives_modules
    group_sizes = global_context.group_sizes
    if len(update_params) != len(update_modules):
        raise Exception('parallel_optimize: number of arguments in update_params does not match number of imported '
                        'submodules.')
    if len(get_features) != len(features_modules):
        raise Exception('parallel_optimize: number of arguments in get_features does not match number of imported '
                        'submodules.')
    if len(features_modules) != len(group_sizes):
        raise Exception('parallel_optimize: number of arguments in group_sizes does not match number of imported '
                        'submodules.')
    module_set = set(update_modules)
    module_set.update(features_modules, objectives_modules)
    global_context.module_set = module_set
    for module_name in module_set:
        m = importlib.import_module(module_name)
        m.config_controller(global_context.export_file_path, output_dir=global_context.output_dir,
                            **global_context.kwargs)
    update_params_funcs = []
    for i, module_name in enumerate(update_modules):
        module = sys.modules[module_name]
        func = getattr(module, update_params[i])
        if not callable(func):
            raise Exception('parallel_optimize: update_params: %s for submodule %s is not a callable function.'
                            % (update_params[i], module_name))
        update_params_funcs.append(func)
    global_context.update_params_funcs = update_params_funcs
    get_features_funcs = []
    for i, module_name in enumerate(features_modules):
        module = sys.modules[module_name]
        func = getattr(module, get_features[i])
        if not callable(func):
            raise Exception('parallel_optimize: get_features: %s for submodule %s is not a callable function.'
                            % (get_features[i], module_name))
        get_features_funcs.append(func)
    global_context.get_features_funcs = get_features_funcs
    get_objectives_funcs = []
    for module_name in objectives_modules:
        module = sys.modules[module_name]
        func = getattr(module, 'get_objectives')
        if not callable(func):
            raise Exception('parallel_optimize: submodule %s does not contain a required callable function '
                            'get_objectives.' % module_name)
        get_objectives_funcs.append(func)
    global_context.get_objectives_funcs = get_objectives_funcs
    sys.stdout.flush()


def setup_client_interface():
    """

    """
    param_gen_name = global_context.param_gen_name
    update_params = global_context.update_params
    get_features = global_context.get_features
    objectives_modules = global_context.objectives_modules
    group_sizes = global_context.group_sizes
    pop_size = global_context.pop_size
    output_dir = global_context.output_dir
    disp = global_context.disp
    module_set = global_context.module_set
    update_params_funcs = global_context.update_params_funcs
    param_names = global_context.param_names
    default_params = global_context.default_params
    export_file_path = global_context.export_file_path

    pc = h.ParallelContext()
    global_context.pc = pc
    num_procs = pc.nhost()
    global_context.num_procs = num_procs

    new_group_sizes = []
    num_blocks = []
    for ind, orig_group_size in enumerate(group_sizes):
        indivs_per_block = min(num_procs / orig_group_size, pop_size)
        if indivs_per_block > 0:
            new_group_size = orig_group_size
            un_utilized = num_procs - (indivs_per_block * new_group_size)
            this_num_blocks = int(math.ceil(float(pop_size) / float(indivs_per_block)))
        else:
            new_group_size = num_procs
            if disp:
                print 'parallel_optimize: stage %i (%s) adjusted group_size to not exceed num_processes: ' \
                      '%i' % (ind, get_features[ind], num_procs)
            blocks_per_indiv = int(math.ceil(float(orig_group_size) / float(num_procs)))
            un_utilized = orig_group_size % new_group_size
            if un_utilized > 0:
                un_utilized = num_procs - un_utilized
            this_num_blocks = blocks_per_indiv * pop_size
        if un_utilized > 0 and disp:
            print 'parallel_optimize: stage %i (%s) has up to %i un-utilized processes per block' % \
                  (ind, get_features[ind], un_utilized)
        new_group_sizes.append(new_group_size)
        num_blocks.append(this_num_blocks)
    group_sizes = new_group_sizes
    global_context.group_sizes = group_sizes
    print 'parallel_optimize: %s; parameter generator: %s; num_processes: %i; population size: %i; group sizes: %s;' \
          ' feature calculators: %s; imported submodules: %s; blocks / generation: %s' % \
          (global_context.optimization_title, param_gen_name, num_procs, pop_size,
           ', '.join(str(i) for i in group_sizes), ', '.join(func_name for func_name in get_features),
           ', '.join(name for name in module_set), ', '.join(str(i) for i in num_blocks))

    #pc.submit(init_engine, module_set, update_params_funcs, param_names, default_params, export_file_path,
    #         output_dir, disp, **global_context.kwargs)
    """
    global_context.c[:].execute('from parallel_optimize import *', block=True)
    if sleep:
        time.sleep(120.)
    global_context.c[:].apply_sync(init_engine, module_set, update_params_funcs, param_names, default_params,
                                   export_file_path, output_dir, disp, **global_context.kwargs)
    """


def update_submodule_params(x):
    """

    :param x: array
    """
    for submodule in global_context.module_set:
        sys.modules[submodule].update_submodule_params(x, sys.modules[submodule].context)


def init_engine_interactive(x, verbose=True):
    """

    :param x: dict
    :param verbose: bool
    """
    x_dict = dict(x)
    x_array = param_dict_to_array(x_dict, global_context.param_names)
    module_set = global_context.module_set
    update_params_funcs = global_context.update_params_funcs
    param_names = global_context.param_names
    default_params = global_context.default_params
    export_file_path = global_context.export_file_path
    output_dir = global_context.output_dir
    disp = global_context.disp
    global_context.kwargs['verbose'] = verbose
    """
    init_engine(module_set, update_params_funcs, param_names, default_params, export_file_path, output_dir, disp,
                **global_context.kwargs)
    """
    update_submodule_params(x_array)
    global_context.x_dict = x_dict
    global_context.x_array = x_array


"""
def init_engine(module_set, update_params_funcs, param_names, default_params, export_file_path, output_dir, disp,
                **kwargs):


    :param module_set: set of str (submodule names)
    :param update_params_funcs: list of callable
    :param param_names: list of str
    :param default_params: dict
    :param export_file_path: str (path)
    :param output_dir: str (dir path)
    :param disp: bool
    :param kwargs: dict

    global temp_output_path
    temp_output_path = '%s/parallel_optimize_temp_output_%s_pid%i.hdf5' % \
                       (output_dir, datetime.datetime.today().strftime('%m%d%Y%H%M'), os.getpid())
    for module_name in module_set:
        m = importlib.import_module(module_name)
        config_func = getattr(m, 'config_engine')
        if not callable(config_func):
            raise Exception('parallel_optimize: init_engine: submodule: %s does not contain required callable: '
                            'config_engine' % module_name)
        else:
            config_func(update_params_funcs, param_names, default_params, temp_output_path, export_file_path,
                        output_dir,
                        disp, **kwargs)
    sys.stdout.flush()
"""


def run_optimization():
    """

    """
    for ind, generation in enumerate(param_gen_instance()):
        if (ind > 0) and (ind % global_context.path_length == 0):
            param_gen_instance.storage.save(global_context.storage_file_path, n=global_context.path_length)
        features, objectives = get_all_features(generation)
        param_gen_instance.update_population(features, objectives)
    param_gen_instance.storage.save(global_context.storage_file_path, n=global_context.path_length)


def get_all_features(generation, export=False):
    """

    :param generation: list of arr
    :param export: bool (for exporting voltage traces)
    :return: tuple of list of dict
    """
    group_sizes = global_context.group_sizes
    disp = global_context.disp
    pop_ids = range(len(generation))
    results = []
    curr_generation = {pop_id: generation[pop_id] for pop_id in pop_ids}
    features_dict = {pop_id: {} for pop_id in pop_ids}
    for ind in xrange(len(global_context.get_features_funcs)):
        next_generation = {}
        this_group_size = min(global_context.num_procs, group_sizes[ind])
        global_context.pc.subworlds(this_group_size) #Need to ensure that total number of processors is a multiple of this_group_size!
        """
        usable_procs = global_context.num_procs - (global_context.num_procs % this_group_size)
        client_ranges = [range(start, start + this_group_size) for start in xrange(0, usable_procs, this_group_size)]
        """
        feature_function = global_context.get_features_funcs[ind]
        indivs = [{'pop_id': pop_id, 'x': curr_generation[pop_id],
                   'features': features_dict[pop_id]} for pop_id in curr_generation]
        feature_function(indivs, export)

        """"""
        while len(indivs) > 0 or len(results) > 0:
            num_groups = min(len(client_ranges), len(indivs))
            if num_groups > 0:
                results.extend(map(feature_function, [indivs.pop(0) for i in xrange(num_groups)],
                                   [global_context.c] * num_groups, [client_ranges.pop(0) for i in xrange(num_groups)],
                                   [export] * num_groups))
            ready_results_list = [this_result for this_result in results if this_result['async_result'].ready()]
            if len(ready_results_list) > 0:
                for this_result in ready_results_list:
                    client_ranges.append(this_result['client_range'])
                    if disp:
                        flush_engine_buffer(this_result['async_result'])
                    computed_result_list = this_result['async_result'].get()
                    if None in computed_result_list:
                        if disp:
                            print 'Individual: %i, failed %s in %.2f s' % (this_result['pop_id'],
                                                                           global_context.get_features[ind],
                                                                           this_result['async_result'].wall_time)
                            sys.stdout.flush()
                        features_dict[this_result['pop_id']] = None
                    else:
                        next_generation[this_result['pop_id']] = generation[this_result['pop_id']]
                        if disp:
                            print 'Individual: %i, computing %s took %.2f s' % (this_result['pop_id'],
                                                                                global_context.get_features[ind],
                                                                                this_result['async_result'].wall_time)
                            sys.stdout.flush()
                        if 'filter_features' in this_result:
                            local_time = time.time()
                            filter_features_func = this_result['filter_features']
                            if not callable(filter_features_func):
                                raise Exception('parallel_optimize: filter_features function %s is not callable' %
                                                filter_features_func)
                            new_features = filter_features_func(computed_result_list,
                                                                features_dict[this_result['pop_id']],
                                                                global_context.target_val, global_context.target_range,
                                                                export)
                            if disp:
                                print 'Individual: %i, filtering features %s took %.2f s' % \
                                      (this_result['pop_id'], filter_features_func.__name__, time.time() - local_time)
                                sys.stdout.flush()
                        else:
                            new_features = {key: value for result_dict in computed_result_list for key, value in
                                            result_dict.iteritems()}
                        features_dict[this_result['pop_id']].update(new_features)
                    results.remove(this_result)
                    sys.stdout.flush()
            else:
                time.sleep(1.)
        curr_generation = next_generation
    features = [features_dict[pop_id] for pop_id in pop_ids]
    objectives = []
    for i, this_features in enumerate(features):
        if this_features is None:
            this_objectives = None
        else:
            this_objectives = {}
            for j, objective_function in enumerate(global_context.get_objectives_funcs):
                new_features, new_objectives = objective_function(this_features, global_context.target_val,
                                                                  global_context.target_range)
                features[i] = new_features
                this_objectives.update(new_objectives)
        objectives.append(this_objectives)
    return features, objectives


def export_traces(x, export_file_path=None, discard=True):
    """
    Run simulations on the engines with the given parameter values, have the engines export their results to .hdf5,
    and then read in and plot the results.

    :param x: array
    :param export_file_path: str
    :param discard: bool
    """
    if export_file_path is not None:
        global_context.export_file_path = export_file_path
    else:
        export_file_path = global_context.export_file_path
    exported_features, exported_objectives = get_all_features([x], export=True)
    temp_output_path_list = [temp_output_path for temp_output_path in global_context.c[:]['temp_output_path'] if
                             os.path.isfile(temp_output_path)]
    combine_hdf5_file_paths(temp_output_path_list, export_file_path)
    if discard:
        for temp_output_path in temp_output_path_list:
            os.remove(temp_output_path)
    print 'parallel_optimize: exported output to %s' % export_file_path
    sys.stdout.flush()
    return exported_features, exported_objectives


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_filename) != -1, sys.argv) + 1):])
