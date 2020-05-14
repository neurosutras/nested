"""
Nested parallel multi-objective optimization.

Inspired by scipy.optimize.basinhopping and emoo, nested.optimize provides a parallel computing-compatible interface for
multi-objective parameter optimization. We have implemented the following unique features:
 - Support for specifying absolute and/or relative parameter bounds.
 - Order of magnitude discovery. Initial search occurs in log space for parameters with bounds that span > 2 orders
 of magnitude. As step size decreases over iterations, search converts to linear.
 - Works interchangeably with a variety of parallel frameworks, including ipyparallel, mpi4py.futures, or the NEURON
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
@click.option("--param-gen", type=str, default='PopulationAnnealing')  # "Sobol" and "Pregenerated" also accepted
@click.option("--hot-start", is_flag=True)
@click.option("--storage-file-path", type=str, default=None)
@click.option("--param-file-path", type=str, default=None)
@click.option("--x0-key", type=str, default=None)
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--label", type=str, default=None)
@click.option("--disp", is_flag=True)
@click.option("--interactive", is_flag=True)
@click.pass_context
def main(cli, config_file_path, param_gen, hot_start, storage_file_path, param_file_path, x0_key, output_dir, label,
         disp, interactive):
    """
    :param cli: :class:'click.Context': used to process/pass through unknown click arguments
    :param config_file_path: str (path)
    :param param_gen: str (must refer to callable in globals())
    :param hot_start: bool
    :param storage_file_path: str (path)
    :param param_file_path: str (path)
    :param x0_key: str
    :param output_dir: str
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
    try:
        init_optimize_controller_context(**kwargs)
        start_time = time.time()

        context.interface.apply(init_worker_contexts, context.sources, context.update_context_funcs,
                                context.param_names, context.default_params, context.feature_names,
                                context.objective_names, context.target_val, context.target_range, context.output_dir,
                                context.disp, optimization_title=context.optimization_title, label=context.label,
                                **context.kwargs)
        if disp:
            print('nested.optimize: worker initialization took %.2f s' % (time.time() - start_time))
        sys.stdout.flush()

        context.param_gen_instance = context.ParamGenClass(
            param_names=context.param_names, feature_names=context.feature_names,
            objective_names=context.objective_names, x0=context.x0_array, bounds=context.bounds,
            rel_bounds=context.rel_bounds, disp=disp, hot_start=hot_start,
            storage_file_path=context.storage_file_path, config_file_path=context.config_file_path,
            **context.kwargs)
        optimize()
        context.storage = context.param_gen_instance.storage
        if not context.storage.survivors or not context.storage.survivors[-1]:
            raise RuntimeError('nested.optimize: all models failed to compute required features or objectives')
        context.report = OptimizationReport(storage=context.storage)
        context.best_indiv = context.report.survivors[0]
        context.x_array = context.best_indiv.x
        context.x_dict = param_array_to_dict(context.x_array, context.storage.param_names)
        context.features = param_array_to_dict(context.best_indiv.features, context.feature_names)
        context.objectives = param_array_to_dict(context.best_indiv.objectives, context.objective_names)

        if disp:
            print('best model_id: %i' % context.best_indiv.model_id)
            print('params:')
            pprint.pprint(context.x_dict)
            print('features:')
            pprint.pprint(context.features)
            print('objectives:')
            pprint.pprint(context.objectives)
        sys.stdout.flush()
        time.sleep(1.)

        if not context.interactive:
            context.interface.stop()

    except Exception as e:
        print('nested.optimize: encountered Exception')
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        time.sleep(1.)
        context.interface.stop()
        raise e


def optimize():
    """

    """
    for generation, model_ids in context.param_gen_instance():
        features, objectives = evaluate_population(context, generation, model_ids)
        context.param_gen_instance.update_population(features, objectives)
        del features
        del objectives
    for shutdown_func in context.shutdown_worker_funcs:
        context.interface.apply(shutdown_func)


def evaluate_population(context, population, model_ids=None, export=False):
    """
    The instructions for computing features and objectives specified in the config_file_path are now followed for each
    individual member of a population of parameter arrays (models). If any compute_features or filter_feature function
    returns an empty dict, or a dict that contains the key 'failed', that member of the population is completely removed
    from any further computation. This frees resources for remaining individuals. If any dictionary of features or
    objectives does not contain the full set of expected items, the param_gen_instance will mark those models as failed
    when update_population is called.
    :param context: :class:'Context'
    :param population: list of arr
    :param model_ids: list of str
    :param export: bool; whether to export data to file during model evaluation
    :return: tuple of list of dict
    """
    if model_ids is None:
        working_model_ids = list(range(len(population)))
    else:
        working_model_ids = list(model_ids)
    if len(set(working_model_ids)) != len(population):
        raise RuntimeError('nested.optimize: evaluate_population: provided model_ids must be unique')
    orig_model_ids = list(working_model_ids)
    params_pop_dict = dict(zip(working_model_ids, population))
    features_pop_dict = {model_id: dict() for model_id in working_model_ids}
    objectives_pop_dict = {model_id: dict() for model_id in working_model_ids}
    for stage in context.stages:
        if not working_model_ids:
            break
        params_pop_list = [params_pop_dict[model_id] for model_id in working_model_ids]
        if 'args' in stage:
            group_size = len(stage['args'][0])
            args_population = [stage['args'] for model_id in working_model_ids]
        elif 'get_args_static_func' in stage:
            stage['args'] = context.interface.execute(stage['get_args_static_func'])
            group_size = len(stage['args'][0])
            args_population = [stage['args'] for model_id in working_model_ids]
        elif 'get_args_dynamic_func' in stage:
            features_pop_list = [features_pop_dict[model_id] for model_id in working_model_ids]
            args_population = context.interface.map_sync(stage['get_args_dynamic_func'], params_pop_list,
                                                         features_pop_list)
            group_size = len(args_population[0][0])
        else:
            args_population = [[] for model_id in working_model_ids]
            group_size = 1
        if 'shared_features' in stage:
            for model_id in working_model_ids:
                features_pop_dict[model_id].update(stage['shared_features'])
        elif 'compute_features_shared_func' in stage:
            args = args_population[0]
            this_x = params_pop_list[0]
            this_model_id = 'shared'
            sequences = [[this_x] * group_size] + args + [[this_model_id] * group_size] + [[export] * group_size]
            primitives = context.interface.map_sync(stage['compute_features_shared_func'], *sequences)
            for features_dict in primitives:
                if not features_dict or 'failed' in features_dict:
                    raise RuntimeError('nested.optimize: compute_features_shared function: %s failed' %
                                       stage['compute_features_shared_func'])
            if 'filter_features_func' in stage:
                this_shared_features = context.interface.execute(
                    stage['filter_features_func'], primitives, {}, this_model_id, export)
                if not this_shared_features or 'failed' in this_shared_features:
                    raise RuntimeError('nested.optimize: shared filter_features function: %s failed' %
                                       stage['filter_features_func'])
            else:
                this_shared_features = dict()
                for features_dict in primitives:
                    this_shared_features.update(features_dict)
            del primitives
            stage['shared_features'] = this_shared_features
            for model_id in working_model_ids:
                features_pop_dict[model_id].update(stage['shared_features'])
            del this_shared_features
        else:
            pending = []
            for this_x, args, this_model_id in zip(params_pop_list, args_population, working_model_ids):
                sequences = [[this_x] * group_size] + args + [[this_model_id] * group_size] + [[export] * group_size]
                pending.append(context.interface.map_async(stage['compute_features_func'], *sequences))
            while not all(result.ready(wait=0.1) for result in pending):
                time.sleep(0.1)
            temp_model_ids = list(working_model_ids)
            primitives_pop_dict = {}
            for model_id, result in zip(temp_model_ids, pending):
                this_primitives = result.get()
                for this_features_dict in this_primitives:
                    if not this_features_dict or 'failed' in this_features_dict:
                        working_model_ids.remove(model_id)
                        break
                else:
                    primitives_pop_dict[model_id] = this_primitives
            del pending
            if 'filter_features_func' in stage:
                primitives_pop_list = []
                features_pop_list = []
                for model_id in working_model_ids:
                    primitives_pop_list.append(primitives_pop_dict[model_id])
                    features_pop_list.append(features_pop_dict[model_id])
                new_features_pop_list = context.interface.map_sync(
                    stage['filter_features_func'], primitives_pop_list, features_pop_list, working_model_ids,
                    [export] * len(working_model_ids))
                del primitives_pop_list
                del features_pop_list
                temp_model_ids = list(working_model_ids)
                for model_id, this_features_dict in zip(temp_model_ids, new_features_pop_list):
                    if not this_features_dict or 'failed' in this_features_dict:
                        working_model_ids.remove(model_id)
                    else:
                        features_pop_dict[model_id].update(this_features_dict)
                del new_features_pop_list
            else:
                for model_id in working_model_ids:
                    for this_features_dict in primitives_pop_dict[model_id]:
                        features_pop_dict[model_id].update(this_features_dict)
            del primitives_pop_dict
            del temp_model_ids
        if 'synchronize_func' in stage:
            context.interface.apply(stage['synchronize_func'])
    for get_objectives_func in context.get_objectives_funcs:
        features_pop_list = [features_pop_dict[model_id] for model_id in working_model_ids]
        result_pop_list = context.interface.map_sync(get_objectives_func, features_pop_list, working_model_ids,
                                                [export] * len(working_model_ids))
        del features_pop_list
        temp_model_ids = list(working_model_ids)
        for model_id, (this_features, this_objectives) in zip(temp_model_ids, result_pop_list):
            if not this_objectives or 'failed' in this_objectives or 'failed' in this_features:
                working_model_ids.remove(model_id)
            else:
                features_pop_dict[model_id].update(this_features)
                objectives_pop_dict[model_id].update(this_objectives)
        del result_pop_list
        del temp_model_ids
        if not working_model_ids:
            if context.disp:
                print('nested.optimize: all models failed to compute required features or objectives')
                sys.stdout.flush()
            break
    sys.stdout.flush()
    features_pop_list = [features_pop_dict[model_id] for model_id in orig_model_ids]
    objectives_pop_list = [objectives_pop_dict[model_id] for model_id in orig_model_ids]
    for reset_func in context.reset_worker_funcs:
        context.interface.apply(reset_func)

    return features_pop_list, objectives_pop_list


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
