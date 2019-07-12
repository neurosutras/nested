"""
Nested parallel multi-objective optimization.

Inspired by scipy.optimize.basinhopping and emoo, nested.optimize provides a parallel computing-compatible interface for
multi-objective parameter optimization. We have implemented the following unique features:
 - Support for specifying absolute and/or relative parameter bounds.
 - Order of magnitude discovery. Initial search occurs in log space for parameters with bounds that span > 2 orders
 of magnitude. As step size decreases over iterations, search converts to linear.
 - Works seamlessly with a variety of parallel frameworks, including ipyparallel, mpi4py.futures, or the NEURON
 simulator's MPI-based ParallelContext bulletin board.
 - Algorithm-specific arguments configuring multi-objective evaluation, ranking, and selection can be specified via the
 command line, and are passed forward to the specified parameter generator/optimizer.
 - Convenient interface for storage, export (to .hdf5), and visualization of optimization intermediates.
 - Capable of "hot starting" from a file in case optimization is interrupted midway.

To run, put the directory containing the nested repository into $PYTHONPATH.
From the directory that contains the custom scripts required for your optimization, execute nested.optimize as a module
as follows:
To use with NEURON's ParallelContext backend with N processes:
mpirun -n N python -m nested.optimize --config-file-path=$PATH_TO_CONFIG_YAML --framework=pc

To use with ipyparallel:
ipcluster start -n N &
# wait until engines are ready
python -m nested.optimize --config-file-path=$PATH_TO_CONFIG_YAML --framework=ipyp
"""
__author__ = 'Aaron D. Milstein and Grace Ng'
from nested.parallel import *
from nested.optimize_utils import *
import click

try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass


context = Context()


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True,))
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None)
@click.option("--param-gen", type=str, default='PopulationAnnealing')
@click.option("--analyze", is_flag=True)
@click.option("--hot-start", is_flag=True)
@click.option("--storage-file-path", type=str, default=None)
@click.option("--export", is_flag=True)
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--disp", is_flag=True)
@click.option("--interactive", is_flag=True)
@click.pass_context
def main(cli, config_file_path, param_gen, analyze, hot_start, storage_file_path, export, output_dir, export_file_path,
         label, disp, interactive):
    """
    :param cli: :class:'click.Context': used to process/pass through unknown click arguments
    :param config_file_path: str (path)
    :param param_gen: str (must refer to callable in globals())
    :param analyze: bool
    :param hot_start: bool
    :param storage_file_path: str
    :param export: bool
    :param output_dir: str
    :param export_file_path: str
    :param label: str
    :param disp: bool
    :param interactive: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    kwargs = get_unknown_click_arg_dict(cli.args)
    context.interface = get_parallel_interface(source_file=__file__, source_package=__package__, **kwargs)
    context.interface.start(disp=disp)
    context.interface.ensure_controller()
    init_controller_context(**kwargs)
    context.interface.apply(init_worker_contexts, context.sources, context.update_context_funcs, context.param_names,
                            context.default_params, context.feature_names, context.objective_names, context.target_val,
                            context.target_range, context.export_file_path, context.output_dir, context.disp,
                            optimization_title=context.optimization_title, label=context.label, **context.kwargs)

    sys.stdout.flush()
    try:
        if not analyze:
            context.param_gen_instance = context.ParamGenClass(
                param_names=context.param_names, feature_names=context.feature_names,
                objective_names=context.objective_names, x0=context.x0_array, bounds=context.bounds,
                rel_bounds=context.rel_bounds, disp=disp, hot_start=hot_start,
                storage_file_path=context.storage_file_path, **context.kwargs)
            optimize()
            context.storage = context.param_gen_instance.storage
            context.report = OptimizationReport(storage=context.storage)
            context.best_indiv = context.report.survivors[0]
            context.x_array = context.best_indiv.x
            context.x_dict = param_array_to_dict(context.x_array, context.storage.param_names)
            context.features = param_array_to_dict(context.best_indiv.features, context.feature_names)
            context.objectives = param_array_to_dict(context.best_indiv.objectives, context.objective_names)
        else:
            if context.storage_file_path is not None and os.path.isfile(context.storage_file_path):
                context.report = OptimizationReport(file_path=context.storage_file_path)
                context.best_indiv = context.report.survivors[0]
                print('nested.optimize: analysis mode: best params loaded from history path: %s' %
                      context.storage_file_path)
                context.x_array = context.best_indiv.x
                context.x_dict = param_array_to_dict(context.x_array, context.storage.param_names)
                context.features = param_array_to_dict(context.best_indiv.features, context.feature_names)
                context.objectives = param_array_to_dict(context.best_indiv.objectives, context.objective_names)
                if disp:
                    print('params (loaded from history):')
                    pprint.pprint(context.x_dict)
                    print('features (loaded from history):')
                    pprint.pprint(context.features)
                    print('objectives (loaded from history):')
                    pprint.pprint(context.objectives)
                    sys.stdout.flush()
            else:
                print('nested.optimize: no optimization history loaded; loading initial params')
                context.x_dict = context.x0_dict
                context.x_array = context.x0_array
            if not export:
                start_time = time.time()
                features, objectives = evaluate_population([context.x_array])
                for shutdown_func in context.shutdown_worker_funcs:
                    context.interface.apply(shutdown_func)
                if disp:
                    print('nested.optimize: evaluating individual took %.2f s' % (time.time() - start_time))
                if not (all([feature_name in features[0] for feature_name in context.feature_names]) and
                        all([objective_name in objectives[0] for objective_name in context.objective_names])):
                    if disp:
                        print('nested.optimize: model failed')
                    context.features = features[0]
                    context.objectives = objectives[0]
                else:
                    context.features = {key: features[0][key] for key in context.feature_names}
                    context.objectives = {key: objectives[0][key] for key in context.objective_names}
            context.interface.apply(update_source_contexts, context.x_array)
        sys.stdout.flush()
        if export:
            context.features, context.objectives, context.export_file_path = export_intermediates(context.x_array)
            for shutdown_func in context.shutdown_worker_funcs:
                context.interface.apply(shutdown_func)
        if disp:
            print('params:')
            pprint.pprint(context.x_dict)
            print('features:')
            pprint.pprint(context.features)
            print('objectives:')
            pprint.pprint(context.objectives)
            sys.stdout.flush()
        if not context.interactive:
            context.interface.stop()
    except Exception as e:
        print('nested.optimize: encountered Exception')
        traceback.print_tb(sys.exc_info()[2])
        sys.stdout.flush()
        time.sleep(1.)
        context.interface.stop()
        raise e


def optimize():
    """

    """
    for generation in context.param_gen_instance():
        features, objectives = evaluate_population(generation)
        context.param_gen_instance.update_population(features, objectives)
        del features
        del objectives
    for shutdown_func in context.shutdown_worker_funcs:
        context.interface.apply(shutdown_func)


def evaluate_population(population, export=False):
    """
    The instructions for computing features and objectives specified in the config_file_path are now followed for each
    individual member of a population of parameter arrays (models). If any compute_features or filter_feature function
    returns an empty dict, or a dict that contains the key 'failed', that member of the population is completely removed
    from any further computation. This frees resources for remaining individuals. If any dictionary of features or
    objectives does not contain the full set of expected items, the param_gen_instance will mark those models as failed
    when update_population is called.
    :param population: list of arr
    :param export: bool; whether to export model intermediates
    :return: tuple of list of dict
    """
    params_pop_dict = dict(enumerate(population))
    pop_ids = list(range(len(population)))
    features_pop_dict = {pop_id: dict() for pop_id in pop_ids}
    objectives_pop_dict = {pop_id: dict() for pop_id in pop_ids}
    for stage in context.stages:
        params_pop_list = [params_pop_dict[pop_id] for pop_id in pop_ids]
        if 'args' in stage:
            group_size = len(stage['args'][0])
            args_population = [stage['args'] for pop_id in pop_ids]
        elif 'get_args_static_func' in stage:
            stage['args'] = context.interface.execute(stage['get_args_static_func'])
            group_size = len(stage['args'][0])
            args_population = [stage['args'] for pop_id in pop_ids]
        elif 'get_args_dynamic_func' in stage:
            features_pop_list = [features_pop_dict[pop_id] for pop_id in pop_ids]
            args_population = context.interface.map_sync(stage['get_args_dynamic_func'], params_pop_list,
                                                         features_pop_list)
            group_size = len(args_population[0][0])
        else:
            args_population = [[] for pop_id in pop_ids]
            group_size = 1
        if 'shared_features' in stage:
            for pop_id in pop_ids:
                features_pop_dict[pop_id].update(stage['shared_features'])
        elif 'compute_features_shared_func' in stage:
            args = args_population[0]
            this_x = params_pop_list[0]
            sequences = [[this_x] * group_size] + args + [[export] * group_size]
            results_list = context.interface.map_sync(stage['compute_features_shared_func'], *sequences)
            if 'filter_features_func' in stage:
                this_shared_features = \
                    context.interface.execute(stage['filter_features_func'], results_list, {}, export)
            else:
                this_shared_features = dict()
                for features_dict in results_list:
                    this_shared_features.update(features_dict)
            if not this_shared_features or 'failed' in this_shared_features:
                raise RuntimeError('nested.optimize: compute_features_shared function: %s failed' %
                                   stage['compute_features_shared'])
            stage['shared_features'] = this_shared_features
            for pop_id in pop_ids:
                features_pop_dict[pop_id].update(stage['shared_features'])
            del this_shared_features
        else:
            pending = []
            for this_x, args in zip(params_pop_list, args_population):
                sequences = [[this_x] * group_size] + args + [[export] * group_size]
                pending.append(context.interface.map_async(stage['compute_features_func'], *sequences))
            while not all(result.ready(wait=0.1) for result in pending):
                time.sleep(0.1)
            primitives = [result.get() for result in pending]
            del pending
            if 'filter_features_func' in stage:
                features_pop_list = [features_pop_dict[pop_id] for pop_id in pop_ids]
                new_features = context.interface.map_sync(stage['filter_features_func'], primitives, features_pop_list,
                                                          [export] * len(pop_ids))
                del features_pop_list
                for pop_id, this_features in zip(pop_ids, new_features):
                    if not this_features:
                        this_features = {'failed': True}
                    features_pop_dict[pop_id].update(this_features)
                del new_features
            else:
                for pop_id, results_list in zip(pop_ids, primitives):
                    this_features = \
                        {key: value for features_dict in results_list for key, value in viewitems(features_dict)}
                    if not this_features:
                        this_features = {'failed': True}
                    features_pop_dict[pop_id].update(this_features)
            del primitives
            temp_pop_ids = list(pop_ids)
            for pop_id in temp_pop_ids:
                if not features_pop_dict[pop_id] or 'failed' in features_pop_dict[pop_id]:
                    pop_ids.remove(pop_id)
            del temp_pop_ids
        if 'synchronize_func' in stage:
            context.interface.apply(stage['synchronize_func'])
    for get_objectives_func in context.get_objectives_funcs:
        temp_pop_ids = list(pop_ids)
        features_pop_list = [features_pop_dict[pop_id] for pop_id in pop_ids]
        primitives = context.interface.map_sync(get_objectives_func, features_pop_list, [export] * len(pop_ids))
        del features_pop_list
        for pop_id, this_result in zip(temp_pop_ids, primitives):
            if this_result is None:
                pop_ids.remove(pop_id)
            else:
                this_features, this_objectives = this_result
                features_pop_dict[pop_id].update(this_features)
                objectives_pop_dict[pop_id].update(this_objectives)
        del primitives
        del temp_pop_ids
    sys.stdout.flush()
    features_pop_list = [features_pop_dict[pop_id] for pop_id in range(len(population))]
    objectives_pop_list = [objectives_pop_dict[pop_id] for pop_id in range(len(population))]
    for reset_func in context.reset_worker_funcs:
        context.interface.apply(reset_func)
    return features_pop_list, objectives_pop_list


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
    start_time = time.time()
    """
    TODO: Need a method to label and save model intermediates without having to duplicate model execution at the end of
    optimization.
    temp_output_path_list = [temp_output_path for temp_output_path in
                             context.interface.get('context.temp_output_path') if os.path.isfile(temp_output_path)]
    for temp_output_path in temp_output_path_list:
        os.remove(temp_output_path)
    """
    features, objectives = evaluate_population([x], export=True)
    if context.disp:
        print('nested.optimize: export_intermediates: evaluating individual took %.2f s' % (time.time() - start_time))
    start_time = time.time()
    temp_output_path_list = [temp_output_path for temp_output_path in
                             context.interface.get('context.temp_output_path') if os.path.isfile(temp_output_path)]
    if not temp_output_path_list:
        if context.disp:
            print('nested.optimize: export_intermediates: no data exported - no temp_output_data files found')
    else:
        merge_exported_data(temp_output_path_list, export_file_path, verbose=False)
        if discard:
            for temp_output_path in temp_output_path_list:
                os.remove(temp_output_path)
        if context.disp:
            print('nested.optimize: exporting output to %s took %.2f s' % (export_file_path, time.time() - start_time))
    sys.stdout.flush()
    if not (all([feature_name in features[0] for feature_name in context.feature_names]) and
            all([objective_name in objectives[0] for objective_name in context.objective_names])):
        if context.disp:
            print('nested.optimize: export_intermediates: model failed')
        return features[0], objectives[0], export_file_path
    exported_features = {key: features[0][key] for key in context.feature_names}
    exported_objectives = {key: objectives[0][key] for key in context.objective_names}
    return exported_features, exported_objectives, export_file_path


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
