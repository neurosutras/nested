"""
Nested parallel model evaluation for analysis and data export.

Sets of models to evaluate can be specified in one of two ways:
1) A .yaml file can be specified via the param-file-path command line argument. By default, all models specified in the
.yaml file are evaluated. Alternatively, the -k or --model-key argument can be used to specify a list of models to
evaluate.
2) The .hdf5 file containing the history of a nested optimization can be provided via the storage-file-path argument. By
default, just the "best" model from the optimization history will be evaluated. The --specialists argument can be used
to also evaluate the best performing models for each objective defined in the config-file-path .yaml file.
Alternatively, the -k of --model-key argument can be used to specify a list of models, with accepted keys being either
"best" or the name of a specialist. Additional models can also be specified via the -i or --model-id argument, which
refer to the unique integer ids associated with each model stored in the storage-file-path .hdf5 file.

If the export argument is provided, during model evaluation, data is exported to an .hdf5 file, organized by model
labels.

To run, put the directory containing the nested repository into $PYTHONPATH.
From the directory that contains the custom scripts required for model evaluation, execute nested.analyze as a module
as follows:
To use with NEURON's ParallelContext backend with N processes:
mpirun -n N python -m nested.analyze --config-file-path=$PATH_TO_CONFIG_YAML --framework=pc /
    --param-file-path=$PATH_TO_PARAM_YAML

To use with ipyparallel:
ipcluster start -n N &
# wait until engines are ready
python -m nested.analyze --config-file-path=$PATH_TO_CONFIG_YAML --framework=ipyp --param-file-path=$PATH_TO_PARAM_YAML
"""
__author__ = 'Aaron D. Milstein, Grace Ng, and Prannath Moolchand'
from nested.parallel import *
from nested.optimize import evaluate_population
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
@click.option("--sobol-analysis", is_flag=True)
@click.option("--storage-file-path", type=str, default=None)
@click.option("--param-file-path", type=str, default=None)
@click.option("-k", "--model-key", type=str, multiple=True)
@click.option("-i", "--model-id", type=int, multiple=True)
@click.option("--export", is_flag=True)
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--disp", is_flag=True)
@click.option("--check-config", is_flag=True)
@click.option("--interactive", is_flag=True)
@click.pass_context
def main(cli, config_file_path, sobol_analysis, storage_file_path, param_file_path, model_key, model_id, export,
         output_dir, export_file_path, label, disp, check_config, interactive):
    """
    :param cli: :class:'click.Context': used to process/pass through unknown click arguments
    :param config_file_path: str (path)
    :param sobol_analysis: bool
    :param storage_file_path: str (path)
    :param param_file_path: str (path)
    :param model_key: list of str
    :param model_id: list of int
    :param export: bool
    :param output_dir: str
    :param export_file_path: str
    :param label: str
    :param disp: bool
    :param check_config: bool
    :param interactive: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    kwargs = get_unknown_click_arg_dict(cli.args)
    context.interface = get_parallel_interface(source_file=__file__, source_package=__package__, **kwargs)
    context.interface.start(disp=disp)
    context.interface.ensure_controller()
    init_analyze_controller_context(**kwargs)
    start_time = time.time()
    context.interface.apply(init_worker_contexts, context.sources, context.update_context_funcs, context.param_names,
                            context.default_params, context.feature_names, context.objective_names, context.target_val,
                            context.target_range, context.output_dir, context.disp,
                            optimization_title=context.optimization_title, label=context.label, **context.kwargs)
    if disp:
        print('nested.analyze: worker initialization took %.2f s' % (time.time() - start_time))
    sys.stdout.flush()

    try:
        if check_config:
            context.interface.apply(update_source_contexts, context.x0_array)
        else:
            if len(context.model_id) < 1 and len(context.model_key) < 1:
                param_arrays = [context.x0_array]
                model_ids = [0]
                model_labels = [None]
            else:
                # TODO: get_analyze_model_group
                # param_arrays, model_ids, model_labels = \
                #    get_analyze_model_group(param_file_path=context.param_file_path,
                #                            storage_file_path=context.storage_file_path, model_id=context.model_id,
                #                            model_key=context.model_key)
                print('nested.analyze: not implemented yet')
                sys.stdout.flush()
                return
            print(type(param_arrays), len(param_arrays), type(model_ids), len(model_ids))
            sys.stdout.flush()
            features, objectives = evaluate_population(param_arrays, model_ids=model_ids, export=context.export)
            if context.export:
                merge_exported_data(param_arrays, model_ids, model_labels, features, objectives,
                                    export_file_path=context.export_file_path, output_dir=context.output_dir,
                                    verbose=context.disp)
            for shutdown_func in context.shutdown_worker_funcs:
                context.interface.apply(shutdown_func)
            if disp:
                for i, params in enumerate(param_arrays):
                    this_model_id = model_ids[i]
                    this_model_labels = model_labels[i]
                    this_param_dict = param_array_to_dict(params, context.param_names)
                    this_features = {key: features[0][key] for key in context.feature_names}
                    this_objectives = {key: objectives[0][key] for key in context.objective_names}
                    print('model_id: %i; model_labels: %s' % (this_model_id, this_model_labels))
                    print('params:')
                    pprint.pprint(this_param_dict)
                    print('features:')
                    pprint.pprint(this_features)
                    print('objectives:')
                    pprint.pprint(this_objectives)
                sys.stdout.flush()
        if not context.interactive:
            context.interface.stop()
    except Exception as e:
        print('nested.analyze: encountered Exception')
        traceback.print_exc(file=sys.stdout)
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
        if not pop_ids:
            if context.disp:
                print('nested.optimize: all models failed to compute required features')
                sys.stdout.flush()
            break
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
        if context.comm.rank == 0:
            temp_output_path_list = list(set(temp_output_path_list))  # remove duplicates
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
    
    if os.path.exists(export_file_path):
        f = h5py.File(export_file_path, 'r+')
        f.attrs['param_names'] = np.array(context.param_names, dtype='S32')
        f.attrs['feature_names'] = np.array(context.feature_names, dtype='S32')
        for par, val in zip(context.param_names, x):    
            f.attrs['par_{!s}'.format(par)] = val
        for k, v in exported_features.items():
            f.attrs['fea_{!s}'.format(k)] = v
        f.close()

    return exported_features, exported_objectives, export_file_path


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
