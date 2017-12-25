__author__ = 'Grace Ng and Aaron D. Milstein'
# from function_lib import *
import click
from utils import *
from moopgen import *
import importlib

try:
    import mkl

    mkl.set_num_threads(1)
except:
    pass

script_filename = 'optimize.py'

context = Context()
context.module_default_args = {'framework': 'serial', 'param_gen': 'BGen'}


@click.command()
@click.option("--cluster-id", type=str, default=None)
@click.option("--profile", type=str, default='default')
@click.option("--framework", type=click.Choice(['ipyp', 'mpi', 'pc', 'serial']), default='ipyp')
@click.option("--subworld-size", type=int, default=1)
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None)
@click.option("--param-gen", type=str, default='BGen')
@click.option("--pop-size", type=int, default=100)
@click.option("--wrap-bounds", is_flag=True)
@click.option("--seed", type=int, default=None)
@click.option("--max-iter", type=int, default=50)
@click.option("--path-length", type=int, default=3)
@click.option("--initial-step-size", type=float, default=0.5)
@click.option("--adaptive-step-factor", type=float, default=0.9)
@click.option("--evaluate", type=str, default=None)
@click.option("--select", type=str, default=None)
@click.option("--m0", type=int, default=20)
@click.option("--c0", type=int, default=20)
@click.option("--p_m", type=float, default=0.5)
@click.option("--delta_m", type=int, default=0)
@click.option("--delta_c", type=int, default=0)
@click.option("--mutate_survivors", is_flag=True)
@click.option("--survival-rate", type=float, default=0.2)
@click.option("--sleep", type=int, default=0)
@click.option("--analyze", is_flag=True)
@click.option("--hot-start", is_flag=True)
@click.option("--storage-file-path", type=str, default=None)
@click.option("--export", is_flag=True)
@click.option("--output-dir", type=str, default='data')
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--disp", is_flag=True)
def main(cluster_id, profile, framework, subworld_size, config_file_path, param_gen, pop_size, wrap_bounds, seed,
         max_iter, path_length, initial_step_size, adaptive_step_factor, evaluate, select, m0, c0, p_m, delta_m,
         delta_c, mutate_survivors, survival_rate, sleep, analyze, hot_start, storage_file_path, export, output_dir,
         export_file_path, label, disp):
    """
    :param cluster_id: str (optional, must match cluster-id of running ipcontroller or ipcluster)
    :param profile: str (optional, must match existing ipyparallel profile)
    :param framework: str ('ipyp': ipyparallel, 'mpi': mpi4py.futures, 'pc': neuron.h.ParallelContext and mpi4py)
    :param subworld_size: (for use with 'mpi' or 'pc' frameworks)
    :param config_file_path: str (path)
    :param param_gen: str (must refer to callable in globals())
    :param pop_size: int
    :param wrap_bounds: bool
    :param seed: int
    :param max_iter: int
    :param path_length: int
    :param initial_step_size: float in [0., 1.]  # BGen-specific argument
    :param adaptive_step_factor: float in [0., 1.]  # BGen-specific argument
    :param evaluate: str name of callable that assigns ranks to individuals during optimization
    :param select: str name of callable that select survivors during optimization
    :param m0: int : initial strength of mutation  # EGen-specific argument
    :param c0: int : initial strength of crossover  # EGen-specific argument
    :param p_m: float in [0., 1.] : probability of mutation  # EGen-specific argument
    :param delta_m: int : decrease mutation strength every interval  # EGen-specific argument
    :param delta_c: int : decrease crossover strength every interval  # EGen-specific argument
    :param mutate_survivors: bool  # EGen-specific argument
    :param survival_rate: float
    :param sleep: int
    :param analyze: bool
    :param hot_start: bool
    :param storage_file_path: str
    :param export: bool
    :param output_dir: str
    :param export_file_path: str
    :param label: str
    :param disp: bool
    """
    # requires a global variable context: :class:'Context'

    context.update(locals())
    config_context()

    if framework == 'ipyp':
        context.interface = IpypInterface(cluster_id=context.cluster_id, profile=context.profile, sleep=context.sleep)
    elif framework == 'mpi':
        raise NotImplementedError('nested.optimize: interface for mpi4py.futures framework not yet implemented')
    elif framework == 'pc':
        context.interface = ParallelContextInterface(subworld_size=context.subworld_size)
    elif framework == 'serial':
        context.interface = SerialInterface()
    context.interface.apply(init_worker, context.module_set, context.update_params_funcs, context.param_names,
                            context.default_params, context.export_file_path, context.output_dir, context.disp,
                            context.kwargs)
    async_result = context.interface.map_async(sys.modules['parallel_optimize_GC_leak'].compute_Rinp_features, ['soma'] * 2,
                                [context.x0_array] * 2)
    while not async_result.ready():
        pass
    result = async_result.get()
    pprint.pprint(result)
    if not analyze:
        if hot_start:
            context.param_gen_instance = context.ParamGenClass(
                pop_size=pop_size, x0=param_dict_to_array(context.x0, context.param_names),
                bounds=context.bounds, rel_bounds=context.rel_bounds, wrap_bounds=wrap_bounds, seed=seed,
                max_iter=max_iter, adaptive_step_factor=adaptive_step_factor, p_m=p_m, delta_m=delta_m, delta_c=delta_c,
                mutate_survivors=mutate_survivors, survival_rate=survival_rate, disp=disp,
                hot_start=context.storage_file_path, **context.kwargs)
        else:
            context.param_gen_instance = context.ParamGenClass(
                param_names=context.param_names, feature_names=context.feature_names,
                objective_names=context.objective_names, pop_size=pop_size,
                x0=param_dict_to_array(context.x0, context.param_names), bounds=context.bounds,
                rel_bounds=context.rel_bounds, wrap_bounds=wrap_bounds, seed=seed, max_iter=max_iter,
                path_length=path_length, initial_step_size=initial_step_size, m0=m0, c0=c0, p_m=p_m, delta_m=delta_m,
                delta_c=delta_c, mutate_survivors=mutate_survivors, adaptive_step_factor=adaptive_step_factor,
                survival_rate=survival_rate, disp=disp, **context.kwargs)
    if False:
        if True:
            optimize()
            context.storage = context.param_gen_instance.storage
            context.best_indiv = context.storage.get_best(1, 'last')[0]
            context.x_array = context.best_indiv.x
            context.x_dict = param_array_to_dict(context.x_array, context.storage.param_names)
            if disp:
                print 'nested.optimize: best params:'
                pprint.pprint(context.x_dict)
        elif context.storage_file_path is not None and os.path.isfile(context.storage_file_path):
            context.storage = PopulationStorage(file_path=context.storage_file_path)
            print 'nested.optimize: analysis mode: history loaded from path: %s' % context.storage_file_path
            context.best_indiv = context.storage.get_best(1, 'last')[0]
            context.x_array = context.best_indiv.x
            context.x_dict = param_array_to_dict(context.x_array, context.storage.param_names)
            init_interactive()
            if disp:
                print 'nested.optimize: best params:'
                pprint.pprint(context.x_dict)
        else:
            print 'nested.optimize: analysis mode: no optimization history loaded'
            context.x_dict = context.x0_dict
            context.x_array = context.x0_array
            init_interactive()
            if disp:
                print 'nested.optimize: initial params:'
                pprint.pprint(context.x_dict)
        sys.stdout.flush()
        if export:
            context.exported_features, context.exported_objectives = export_traces(context.x_array)
    if not context.analyze:
        try:
            context.interface.stop()
        except:
            pass


def config_context(config_file_path=None, storage_file_path=None, export_file_path=None, param_gen=None, label=None,
                   analyze=None):
    """

    :param config_file_path: str (path)
    :param storage_file_path: str (path)
    :param export_file_path: str (path)
    :param param_gen: str
    :param label: str
    :param analyze: bool
    """
    if config_file_path is not None:
        context.config_file_path = config_file_path
    if 'config_file_path' not in context() or context.config_file_path is None or \
            not os.path.isfile(context.config_file_path):
        raise Exception('nested.optimize: config_file_path specifying optimization parameters is missing or invalid.')
    config_dict = read_from_yaml(context.config_file_path)
    context.param_names = config_dict['param_names']
    if 'default_params' not in config_dict or config_dict['default_params'] is None:
        context.default_params = {}
    else:
        context.default_params = config_dict['default_params']
    for param in context.default_params:
        config_dict['bounds'][param] = (context.default_params[param], context.default_params[param])
    context.bounds = [config_dict['bounds'][key] for key in context.param_names]
    if 'rel_bounds' not in config_dict or config_dict['rel_bounds'] is None:
        context.rel_bounds = None
    else:
        context.rel_bounds = config_dict['rel_bounds']
    if 'x0' not in config_dict or config_dict['x0'] is None:
        context.x0 = None
    else:
        context.x0 = config_dict['x0']
        context.x0_dict = context.x0
        context.x0_array = param_dict_to_array(context.x0_dict, context.param_names)
    context.feature_names = config_dict['feature_names']
    context.objective_names = config_dict['objective_names']
    context.target_val = config_dict['target_val']
    context.target_range = config_dict['target_range']
    context.optimization_title = config_dict['optimization_title']
    context.kwargs = config_dict['kwargs']  # Extra arguments to be passed to imported submodules
    context.update(context.kwargs)

    missing_config = []
    if 'update_params' not in config_dict or config_dict['update_params'] is None:
        context.update_params = []
    else:
        context.update_params = config_dict['update_params']
    if 'update_modules' not in config_dict or config_dict['update_modules'] is None:
        context.update_modules = []
    else:
        context.update_modules = config_dict['update_modules']
    if 'get_features' not in config_dict or config_dict['get_features'] is None:
        missing_config.append('get_features')
    else:
        context.get_features = config_dict['get_features']
    if 'features_modules' not in config_dict or config_dict['features_modules'] is None:
        missing_config.append('features_modules')
    else:
        context.features_modules = config_dict['features_modules']
    if 'objectives_modules' not in config_dict or config_dict['objectives_modules'] is None:
        missing_config.append('objectives_modules')
    else:
        context.objectives_modules = config_dict['objectives_modules']
    if 'group_sizes' not in config_dict or config_dict['group_sizes'] is None:
        missing_config.append('group_sizes')
    else:
        context.group_sizes = config_dict['group_sizes']
    if missing_config:
        raise Exception('nested.optimize: config_file at path: %s is missing the following required fields: %s' %
                        (context.config_file_path, ', '.join(str(field) for field in missing_config)))

    if label is not None:
        context.label = label
    if 'label' not in context() or context.label is None:
        label = ''
    else:
        label = '_' + context.label
    if param_gen is not None:
        context.param_gen = param_gen
    if 'param_gen' not in context():
        context.ParamGenClassName = context.module_default_args['param_gen']
    else:
        context.ParamGenClassName = context.param_gen
    if context.ParamGenClassName not in globals():
        raise Exception('nested.optimize: %s has not been imported, or is not a valid class of parameter '
                        'generator.' % context.ParamGenClassName)
    if storage_file_path is not None:
        context.storage_file_path = storage_file_path
    if 'storage_file_path' not in context() or context.storage_file_path is None:
        context.storage_file_path = '%s/%s_%s%s_%s_optimization_history.hdf5' % \
                                    (context.output_dir, datetime.datetime.today().strftime('%m%d%Y%H%M'),
                                     context.optimization_title, label, context.ParamGenClassName)
    if export_file_path is not None:
        context.export_file_path = export_file_path
    if 'export_file_path' not in context() or context.export_file_path is None:
        context.export_file_path = '%s/%s_%s%s_%s_optimization_exported_output.hdf5' % \
                                   (context.output_dir, datetime.datetime.today().strftime('%m%d%Y%H%M'),
                                    context.optimization_title, label, context.ParamGenClassName)

    # ParamGenClass points to the parameter generator class, while ParamGenClassName points to its name as a string
    context.ParamGenClass = globals()[context.ParamGenClassName]

    if len(context.update_params) != len(context.update_modules):
        raise Exception('nested.optimize: number of arguments in update_params does not match number of imported '
                        'submodules.')
    if len(context.get_features) != len(context.features_modules):
        raise Exception('nested.optimize: number of arguments in get_features does not match number of imported '
                        'submodules.')
    if len(context.features_modules) != len(context.group_sizes):
        raise Exception('nested.optimize: number of arguments in group_sizes does not match number of imported '
                        'submodules.')
    context.module_set = set(context.update_modules)
    context.module_set.update(context.features_modules, context.objectives_modules)
    for module_name in context.module_set:
        m = importlib.import_module(module_name)
        m.config_controller(context.export_file_path, output_dir=context.output_dir, **context.kwargs)
    context.update_params_funcs = []
    for i, module_name in enumerate(context.update_modules):
        module = sys.modules[module_name]
        func = getattr(module, context.update_params[i])
        if not isinstance(func, collections.Callable):
            raise Exception('nested.optimize: update_params: %s for submodule %s is not a callable function.'
                            % (context.update_params[i], module_name))
        context.update_params_funcs.append(func)
    context.get_features_funcs = []
    for i, module_name in enumerate(context.features_modules):
        module = sys.modules[module_name]
        func = getattr(module, context.get_features[i])
        if not isinstance(func, collections.Callable):
            raise Exception('nested.optimize: get_features: %s for submodule %s is not a callable function.'
                            % (context.get_features[i], module_name))
        context.get_features_funcs.append(func)
    context.get_objectives_funcs = []
    for module_name in context.objectives_modules:
        module = sys.modules[module_name]
        func = getattr(module, 'get_objectives')
        if not isinstance(func, collections.Callable):
            raise Exception('nested.optimize: submodule %s does not contain a required callable function '
                            'get_objectives.' % module_name)
        context.get_objectives_funcs.append(func)
    if analyze is not None:
        context.analyze = analyze
    if 'analyze' in context() and context.analyze:
        context.pop_size = 1


def update_submodule_params(x):
    """

    :param x: array
    """
    for submodule in context.module_set:
        sys.modules[submodule].update_submodule_params(x, sys.modules[submodule].context)


def init_interactive(verbose=True):
    """

    :param verbose: bool
    """
    init_worker(context.module_set, context.update_params_funcs, context.param_names, context.default_params,
                context.export_file_path, context.output_dir, context.disp, context.kwargs)
    context.kwargs['verbose'] = verbose
    update_submodule_params(context.x_array)


def init_worker(module_set, update_params_funcs, param_names, default_params, export_file_path, output_dir, disp,
                kwargs):
    """

    :param module_set: set of str (submodule names)
    :param update_params_funcs: list of callable
    :param param_names: list of str
    :param default_params: dict
    :param export_file_path: str (path)
    :param output_dir: str (dir path)
    :param disp: bool
    :param kwargs: dict
    """
    context.temp_output_path = '%s/nested.optimize_temp_output_%s_pid%i.hdf5' % \
                               (output_dir, datetime.datetime.today().strftime('%m%d%Y%H%M'), os.getpid())
    for module_name in module_set:
        m = importlib.import_module(module_name)
        config_func = getattr(m, 'config_engine')
        if not isinstance(config_func, collections.Callable):
            raise Exception('nested.optimize: init_worker: submodule: %s does not contain required callable: '
                            'config_engine' % module_name)
        else:
            config_func(update_params_funcs, param_names, default_params, context.temp_output_path, export_file_path,
                        output_dir, disp, kwargs)
    try:
        context.interface.start(disp=disp)
    except:
        pass
    sys.stdout.flush()


def optimize():
    """

    """
    for ind, generation in enumerate(context.param_gen_instance()):
        if (ind > 0) and (ind % context.path_length == 0):
            context.param_gen_instance.storage.save(context.storage_file_path, n=context.path_length)
        features, objectives = get_all_features(generation)
        context.param_gen_instance.update_population(features, objectives)
    context.param_gen_instance.storage.save(context.storage_file_path, n=context.path_length)


def get_all_features(generation, export=False):
    """

    :param generation: list of arr
    :param export: bool (for exporting voltage traces)
    :return: tuple of list of dict
    """
    group_sizes = context.group_sizes
    disp = context.disp
    pop_ids = range(len(generation))
    results = []
    curr_generation = {pop_id: generation[pop_id] for pop_id in pop_ids}
    features_dict = {pop_id: {} for pop_id in pop_ids}
    for ind in xrange(len(context.get_features_funcs)):
        next_generation = {}
        this_group_size = min(context.num_procs, group_sizes[ind])
        usable_procs = context.num_procs - (context.num_procs % this_group_size)
        client_ranges = [range(start, start + this_group_size) for start in xrange(0, usable_procs, this_group_size)]
        feature_function = context.get_features_funcs[ind]
        indivs = [{'pop_id': pop_id, 'x': curr_generation[pop_id],
                   'features': features_dict[pop_id]} for pop_id in curr_generation]
        while len(indivs) > 0 or len(results) > 0:
            num_groups = min(len(client_ranges), len(indivs))
            if num_groups > 0:
                results.extend(map(feature_function, [indivs.pop(0) for i in xrange(num_groups)],
                                   [context.c] * num_groups, [client_ranges.pop(0) for i in xrange(num_groups)],
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
                                                                           context.get_features[ind],
                                                                           this_result['async_result'].wall_time)
                            sys.stdout.flush()
                        features_dict[this_result['pop_id']] = None
                    else:
                        next_generation[this_result['pop_id']] = generation[this_result['pop_id']]
                        if disp:
                            print 'Individual: %i, computing %s took %.2f s' % (this_result['pop_id'],
                                                                                context.get_features[ind],
                                                                                this_result['async_result'].wall_time)
                            sys.stdout.flush()
                        if 'filter_features' in this_result:
                            local_time = time.time()
                            filter_features_func = this_result['filter_features']
                            if not isinstance(filter_features_func, collections.Callable):
                                raise Exception('nested.optimize: filter_features function %s is not callable' %
                                                filter_features_func)
                            new_features = filter_features_func(computed_result_list,
                                                                features_dict[this_result['pop_id']],
                                                                context.target_val, context.target_range,
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
            for j, objective_function in enumerate(context.get_objectives_funcs):
                new_features, new_objectives = objective_function(this_features, context.target_val,
                                                                  context.target_range)
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
        context.export_file_path = export_file_path
    else:
        export_file_path = context.export_file_path
    exported_features, exported_objectives = get_all_features([x], export=True)
    temp_output_path_list = [temp_output_path for temp_output_path in context.c[:]['temp_output_path'] if
                             os.path.isfile(temp_output_path)]
    combine_hdf5_file_paths(temp_output_path_list, export_file_path)
    if discard:
        for temp_output_path in temp_output_path_list:
            os.remove(temp_output_path)
    print 'nested.optimize: exported output to %s' % export_file_path
    sys.stdout.flush()
    return exported_features, exported_objectives


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_filename) != -1, sys.argv) + 1):])
