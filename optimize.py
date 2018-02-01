"""
Nested parallel multi-objective optimization.

Inspired by scipy.optimize.basinhopping and emoo, nested.optimize provides a parallel computing-compatible interface for
multi-objective parameter optimization. We have implemented the following unique features:
 - Support for specifying absolute and/or relative parameter bounds.
 - Order of magnitude discovery. Initial search occurs in log space for parameters with bounds that span > 2 orders
 of magnitude. As step size decreases over iterations, search converts to linear.
 - Hyper-parameter dynamics, generation of parameters, and multi-objective evaluation, ranking, and selection are kept
 separate from the specifics of the framework used for parallel processing.
 - Convenient interface for storage, visualization, and export (to .hdf5) of intermediates during optimization.
 - Capable of "hot starting" from a file in case optimization is interrupted midway.
 """
__author__ = 'Aaron D. Milstein and Grace Ng'
from nested.utils import *
from nested.parallel import *
from nested.optimize_utils import *
import importlib
import click

try:
    import mkl

    mkl.set_num_threads(1)
except:
    pass


context = Context()
context.module_default_args = {'framework': 'serial', 'param_gen': 'PopulationAnnealing'}


@click.command()
@click.option("--cluster-id", type=str, default=None)
@click.option("--profile", type=str, default='default')
@click.option("--framework", type=click.Choice(['ipyp', 'mpi', 'pc', 'serial']), default='ipyp')
@click.option("--procs-per-worker", type=int, default=1)
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None)
@click.option("--param-gen", type=str, default='PopulationAnnealing')
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
@click.option("--max-fitness", type=int, default=5)
@click.option("--sleep", type=int, default=0)
@click.option("--analyze", is_flag=True)
@click.option("--hot-start", is_flag=True)
@click.option("--storage-file-path", type=str, default=None)
@click.option("--export", is_flag=True)
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--disp", is_flag=True)
def main(cluster_id, profile, framework, procs_per_worker, config_file_path, param_gen, pop_size, wrap_bounds, seed,
         max_iter, path_length, initial_step_size, adaptive_step_factor, evaluate, select, m0, c0, p_m, delta_m,
         delta_c, mutate_survivors, survival_rate, max_fitness, sleep, analyze, hot_start, storage_file_path, export,
         output_dir, export_file_path, label, disp):
    """
    :param cluster_id: str (optional, must match cluster-id of running ipcontroller or ipcluster)
    :param profile: str (optional, must match existing ipyparallel profile)
    :param framework: str ('ipyp': ipyparallel, 'mpi': mpi4py.futures, 'pc': neuron.h.ParallelContext and mpi4py)
    :param procs_per_worker: (for use with 'mpi' or 'pc' frameworks)
    :param config_file_path: str (path)
    :param param_gen: str (must refer to callable in globals())
    :param pop_size: int
    :param wrap_bounds: bool
    :param seed: int
    :param max_iter: int
    :param path_length: int
    :param initial_step_size: float in [0., 1.]  # PopulationAnnealing-specific argument
    :param adaptive_step_factor: float in [0., 1.]  # PopulationAnnealing-specific argument
    :param evaluate: str name of callable that assigns ranks to individuals during optimization
    :param select: str name of callable that select survivors during optimization
    :param m0: int : initial strength of mutation  # Evolution-specific argument
    :param c0: int : initial strength of crossover  # Evolution-specific argument
    :param p_m: float in [0., 1.] : probability of mutation  # Evolution-specific argument
    :param delta_m: int : decrease mutation strength every interval  # Evolution-specific argument
    :param delta_c: int : decrease crossover strength every interval  # Evolution-specific argument
    :param mutate_survivors: bool  # Evolution-specific argument
    :param survival_rate: float
    :param max_fitness: int
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
    if framework == 'ipyp':
        context.interface = IpypInterface(cluster_id=context.cluster_id, profile=context.profile,
                                          procs_per_worker=context.procs_per_worker, sleep=context.sleep,
                                          source_file=__file__, source_package=__package__)
    elif framework == 'mpi':
        raise NotImplementedError('nested.optimize: interface for mpi4py.futures framework not yet implemented')
    elif framework == 'pc':
        context.interface = ParallelContextInterface(procs_per_worker=context.procs_per_worker)
    elif framework == 'serial':
        context.interface = SerialInterface()
    config_context()
    context.interface.apply(init_worker, context.sources, context.update_context_funcs, context.param_names,
                            context.default_params, context.target_val, context.target_range, context.export_file_path,
                            context.output_dir, context.disp, **context.kwargs)
    context.interface.ensure_controller()
    if not analyze:
        if hot_start:
            context.param_gen_instance = context.ParamGenClass(
                pop_size=pop_size, x0=param_dict_to_array(context.x0, context.param_names),
                bounds=context.bounds, rel_bounds=context.rel_bounds, wrap_bounds=wrap_bounds, seed=seed,
                max_iter=max_iter, adaptive_step_factor=adaptive_step_factor, p_m=p_m, delta_m=delta_m, delta_c=delta_c,
                mutate_survivors=mutate_survivors, survival_rate=survival_rate, max_fitness=max_fitness, disp=disp,
                hot_start=hot_start, storage_file_path=context.storage_file_path, **context.kwargs)
        else:
            context.param_gen_instance = context.ParamGenClass(
                param_names=context.param_names, feature_names=context.feature_names,
                objective_names=context.objective_names, pop_size=pop_size,
                x0=param_dict_to_array(context.x0, context.param_names), bounds=context.bounds,
                rel_bounds=context.rel_bounds, wrap_bounds=wrap_bounds, seed=seed, max_iter=max_iter,
                path_length=path_length, initial_step_size=initial_step_size, m0=m0, c0=c0, p_m=p_m, delta_m=delta_m,
                delta_c=delta_c, mutate_survivors=mutate_survivors, adaptive_step_factor=adaptive_step_factor,
                survival_rate=survival_rate, max_fitness=max_fitness, disp=disp, hot_start=hot_start,
                storage_file_path=context.storage_file_path, **context.kwargs)
        optimize()
        context.storage = context.param_gen_instance.storage
        context.best_indiv = context.storage.get_best(1, 'last')[0]
        context.x_array = context.best_indiv.x
        context.x_dict = param_array_to_dict(context.x_array, context.storage.param_names)
        context.features = param_array_to_dict(context.best_indiv.features, context.feature_names)
        context.objectives = param_array_to_dict(context.best_indiv.objectives, context.objective_names)
        if disp:
            print 'nested.optimize: best individual: params:'
            pprint.pprint(context.x_dict)
    elif context.storage_file_path is not None and os.path.isfile(context.storage_file_path):
        context.storage = PopulationStorage(file_path=context.storage_file_path)
        print 'nested.optimize: analysis mode: history loaded from path: %s' % context.storage_file_path
        context.best_indiv = context.storage.get_best(1, 'last')[0]
        context.x_array = context.best_indiv.x
        context.x_dict = param_array_to_dict(context.x_array, context.storage.param_names)
        context.features = param_array_to_dict(context.best_indiv.features, context.feature_names)
        context.objectives = param_array_to_dict(context.best_indiv.objectives, context.objective_names)
        if disp:
            print 'nested.optimize: best individual: params:'
            pprint.pprint(context.x_dict)
        context.interface.apply(update_source_contexts, context.x_array)
    else:
        print 'nested.optimize: analysis mode: no optimization history loaded'
        context.x_dict = context.x0_dict
        context.x_array = context.x0_array
        if not export:
            context.features, context.objectives = evaluate_population([context.x_array])
        if disp:
            print 'nested.optimize: initial params:'
            pprint.pprint(context.x_dict)
        context.interface.apply(update_source_contexts, context.x_array)
    sys.stdout.flush()
    if export:
        context.features, context.objectives, context.export_file_path = export_intermediates(context.x_array)
    if disp:
        print 'features:'
        pprint.pprint(context.features)
        print 'objectives:'
        pprint.pprint(context.objectives)
    if not context.analyze:
        try:
            context.interface.stop()
        except Exception:
            pass


def config_context(config_file_path=None, storage_file_path=None, export_file_path=None, param_gen=None, label=None,
                   analyze=None, output_dir=None):
    """

    :param config_file_path: str (path)
    :param storage_file_path: str (path)
    :param export_file_path: str (path)
    :param param_gen: str
    :param label: str
    :param analyze: bool
    :param output_dir: str (dir)
    """
    if config_file_path is not None:
        context.config_file_path = config_file_path
    if 'config_file_path' not in context() or context.config_file_path is None or \
            not os.path.isfile(context.config_file_path):
        raise Exception('nested.optimize: config_file_path specifying required optimization parameters is missing or '
                        'invalid.')
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
    context.kwargs = config_dict['kwargs']  # Extra arguments to be passed to imported sources
    context.update(context.kwargs)

    missing_config = []
    if 'update_context' not in config_dict or config_dict['update_context'] is None:
        missing_config.append('update_context')
    else:
        context.update_context_dict = config_dict['update_context']
    if 'get_features_stages' not in config_dict or config_dict['get_features_stages'] is None:
        missing_config.append('get_features_stages')
    else:
        context.stages = config_dict['get_features_stages']
    if 'get_objectives' not in config_dict or config_dict['get_objectives'] is None:
        missing_config.append('get_objectives')
    else:
        context.get_objectives_dict = config_dict['get_objectives']
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
    # ParamGenClass points to the parameter generator class, while ParamGenClassName points to its name as a string
    if context.ParamGenClassName not in globals():
        raise Exception('nested.optimize: %s has not been imported, or is not a valid class of parameter '
                        'generator.' % context.ParamGenClassName)
    context.ParamGenClass = globals()[context.ParamGenClassName]
    if output_dir is not None:
        context.output_dir = output_dir
    if 'output_dir' not in context():
        context.output_dir = None
    if context.output_dir is None:
        output_dir_str = ''
    else:
        output_dir_str = context.output_dir + '/'
    if storage_file_path is not None:
        context.storage_file_path = storage_file_path
    if 'storage_file_path' not in context() or context.storage_file_path is None:
        context.storage_file_path = '%s%s_%s%s_%s_optimization_history.hdf5' % \
                                    (output_dir_str, datetime.datetime.today().strftime('%Y%m%d%H%M'),
                                     context.optimization_title, label, context.ParamGenClassName)
    if export_file_path is not None:
        context.export_file_path = export_file_path
    if 'export_file_path' not in context() or context.export_file_path is None:
        context.export_file_path = '%s%s_%s%s_%s_optimization_exported_output.hdf5' % \
                                   (output_dir_str, datetime.datetime.today().strftime('%Y%m%d%H%M'),
                                    context.optimization_title, label, context.ParamGenClassName)

    context.sources = set(context.update_context_dict.keys() + context.get_objectives_dict.keys() +
                          [stage['source'] for stage in context.stages if 'source' in stage])
    for source in context.sources:
        m = importlib.import_module(source)
        m.config_controller(export_file_path=context.export_file_path, output_dir=context.output_dir, **context.kwargs)

    context.update_context_funcs = []
    for source, func_name in context.update_context_dict.iteritems():
        module = sys.modules[source]
        func = getattr(module, func_name)
        if not isinstance(func, collections.Callable):
            raise Exception('nested.optimize: update_context: %s for source: %s is not a callable function.'
                            % (func_name, source))
        context.update_context_funcs.append(func)
    context.group_sizes = []
    for stage in context.stages:
        source = stage['source']
        module = sys.modules[source]
        if 'group_size' in stage and stage['group_size'] is not None:
            context.group_sizes.append(stage['group_size'])
        else:
            context.group_sizes.append(1)
        if 'get_args_static' in stage and stage['get_args_static'] is not None:
            func_name = stage['get_args_static']
            func = getattr(module, func_name)
            if not isinstance(func, collections.Callable):
                raise Exception('nested.optimize: get_args_static: %s for source: %s is not a callable function.'
                                % (func_name, source))
            stage['get_args_static_func'] = func
        elif 'get_args_dynamic' in stage and stage['get_args_dynamic'] is not None:
            func_name = stage['get_args_dynamic']
            func = getattr(module, func_name)
            if not isinstance(func, collections.Callable):
                raise Exception('nested.optimize: get_args_dynamic: %s for source: %s is not a callable function.'
                                % (func_name, source))
            stage['get_args_dynamic_func'] = func
        if 'compute_features' in stage and stage['compute_features'] is not None:
            func_name = stage['compute_features']
            func = getattr(module, func_name)
            if not isinstance(func, collections.Callable):
                raise Exception('nested.optimize: compute_features: %s for source: %s is not a callable function.'
                                % (func_name, source))
            stage['compute_features_func'] = func
        if 'filter_features' in stage and stage['filter_features'] is not None:
            func_name = stage['filter_features']
            func = getattr(module, func_name)
            if not isinstance(func, collections.Callable):
                raise Exception('nested.optimize: filter_features: %s for source: %s is not a callable function.'
                                % (func_name, source))
            stage['filter_features_func'] = func
    context.get_objectives_funcs = []
    for source, func_name in context.get_objectives_dict.iteritems():
        module = sys.modules[source]
        func = getattr(module, func_name)
        if not isinstance(func, collections.Callable):
            raise Exception('nested.optimize: get_objectives: %s for source: %s is not a callable function.'
                            % (func_name, source))
        context.get_objectives_funcs.append(func)
    if analyze is not None:
        context.analyze = analyze
    if 'analyze' in context() and context.analyze:
        context.pop_size = 1


def update_source_contexts(x):
    """

    :param x: array
    """
    for source in context.sources:
        sys.modules[source].update_source_contexts(x, sys.modules[source].context)


def init_worker(sources, update_context_funcs, param_names, default_params, target_val, target_range, export_file_path,
                output_dir, disp, **kwargs):
    """

    :param sources: set of str (source names)
    :param update_context_funcs: list of callable
    :param param_names: list of str
    :param default_params: dict
    :param target_val: dict
    :param target_range: dict
    :param export_file_path: str (path)
    :param output_dir: str (dir path)
    :param disp: bool
    """
    if output_dir is not None:
        context.output_dir = output_dir
    if 'output_dir' not in context():
        context.output_dir = None
    if context.output_dir is None:
        output_dir_str = ''
    else:
        output_dir_str = context.output_dir + '/'
    context.temp_output_path = '%snested_optimize_temp_output_%s_pid%i.hdf5' % \
                               (output_dir_str, datetime.datetime.today().strftime('%Y%m%d%H%M'), os.getpid())
    context.sources = sources
    for source in sources:
        m = importlib.import_module(source)
        config_func = getattr(m, 'config_worker')
        try:
            if 'interface' in context():
                m.context.interface = context.interface
        except Exception:
            print 'nested.optimize: init_worker: interface cannot be referenced by worker with pid: %i' % os.getpid()
        if not isinstance(config_func, collections.Callable):
            raise Exception('nested.optimize: init_worker: source: %s does not contain required callable: '
                            'config_engine' % source)
        else:
            config_func(update_context_funcs, param_names, default_params, target_val, target_range,
                        context.temp_output_path, export_file_path, output_dir, disp, **kwargs)
    try:
        context.interface.start(disp=disp)
    except:
        pass
    sys.stdout.flush()


def optimize():
    """

    """
    for generation in context.param_gen_instance():
        features, objectives = evaluate_population(generation)
        context.param_gen_instance.update_population(features, objectives)


def evaluate_population(population, export=False):
    """

    :param population: list of arr
    :param export: bool (for exporting voltage traces)
    :return: tuple of list of dict
    """
    pop_size = len(population)
    features = [{} for pop_id in xrange(pop_size)]
    objectives = [{} for pop_id in xrange(pop_size)]
    for stage in context.stages:
        if 'args' in stage:
            group_size = len(stage['args'][0])
            args_population = [stage['args'] for pop_id in xrange(pop_size)]
        elif 'get_args_static_func' in stage:
            stage['args'] = stage['get_args_static_func']()
            group_size = len(stage['args'][0])
            args_population = [stage['args'] for pop_id in xrange(pop_size)]
        elif 'get_args_dynamic_func' in stage:
            args_population = context.interface.map_sync(stage['get_args_dynamic_func'], population, features)
            group_size = len(args_population[0][0])
        else:
            args_population = [[] for pop_id in xrange(pop_size)]
            group_size = 1
        pending = []
        for pop_id, args in enumerate(args_population):
            this_x = population[pop_id]
            sequences = [[this_x] * group_size] + args + [[export] * group_size]
            pending.append(context.interface.map_async(stage['compute_features_func'], *sequences))
        while not all(result.ready() for result in pending):
            time.sleep(0.1)
            # pass
        primitives = [result.get() for result in pending]
        del pending
        gc.collect()
        if 'filter_features_func' in stage:
            new_features = context.interface.map_sync(stage['filter_features_func'], primitives, features,
                                                    [export] * pop_size)
            for pop_id, this_features in enumerate(new_features):
                features[pop_id].update(this_features)
            del new_features
            gc.collect()
        else:
            for pop_id, results_list in enumerate(primitives):
                this_features = \
                    {key: value for features_dict in results_list for key, value in features_dict.iteritems()}
                features[pop_id].update(this_features)
    del primitives
    gc.collect()
    for get_objectives_func in context.get_objectives_funcs:
        primitives = context.interface.map_sync(get_objectives_func, features)
        for pop_id, (this_features, this_objectives) in enumerate(primitives):
            features[pop_id].update(this_features)
            objectives[pop_id].update(this_objectives)
    del primitives
    gc.collect()
    return features, objectives


def export_intermediates(x, export_file_path=None, discard=True):
    """
    During calculation of features and objectives, source methods may respond to the export flag by appending
    intermediates like simulation output to separate .hdf5 files on each process. This method evaluates a single
    parameter array and merges the resulting .hdf5 files.
    :param x: array
    :param export_file_path: str
    :param discard: bool
    """
    if export_file_path is not None:
        context.export_file_path = export_file_path
    else:
        export_file_path = context.export_file_path
    exported_features, exported_objectives = evaluate_population([x], export=True)
    temp_output_path_list = [temp_output_path for temp_output_path in
                             context.interface.get('context.temp_output_path') if os.path.isfile(temp_output_path)]
    merge_hdf5_files(temp_output_path_list, export_file_path, verbose=False)
    if discard:
        for temp_output_path in temp_output_path_list:
            os.remove(temp_output_path)
    print 'nested.optimize: exported output to %s' % export_file_path
    sys.stdout.flush()
    return exported_features, exported_objectives, export_file_path


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
