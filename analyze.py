"""
Nested parallel model evaluation for analysis and data export.

Sets of models to evaluate can be specified in one of two ways:
1) A .yaml file can be specified via the param-file-path command line argument. By default, all models specified in the
.yaml file are evaluated. Alternatively, the -k or --model-key argument can be used to specify a list of models to
evaluate.
2) The .hdf5 file containing the history of a nested optimization can be provided via the storage-file-path argument. By
default, just the 'best' model from the optimization history will be evaluated. The -k or --model-key argument can be
used to specify a list of models, with accepted keys being either 'best', or the name of an objective to evaluate the
'specialist' model for that objective. 'all' can be provided to evaluate both the 'best' model and all specialist models.
Additional models can also be specified by the integer model_id via the -i or --model-id argument.

If the --export argument is provided, data is exported to an .hdf5 file, organized by model labels.

To run, put the directory containing the nested repository into $PYTHONPATH.
From the directory that contains the custom scripts required for model evaluation, execute nested.analyze as a module
as follows:
To use with a single process:
python -m nested.analyze --config-file-path=$PATH_TO_CONFIG_YAML --framework=serial

To use with NEURON's ParallelContext backend with N processes:
mpirun -n N python -m nested.analyze --config-file-path=$PATH_TO_CONFIG_YAML --framework=pc

To use with mpi4py's future backend with 1 controller and (N - 1) workers:
mpirun -n N python -m mpi4py.futures -m nested.analyze --config-file-path=$PATH_TO_CONFIG_YAML --framework=mpi

To use with ipyparallel backend with N workers:
ipcluster start -n N &
# wait until engines are ready
python -m nested.analyze --config-file-path=$PATH_TO_CONFIG_YAML --framework=ipyp
"""
__author__ = 'Aaron D. Milstein, Grace Ng, and Prannath Moolchand'
from nested.optimize_utils import *
from nested.parallel import *
from nested.optimize import evaluate_population
import click

"""
try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass
"""


context = Context()


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True,))
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None)
@click.option("--sobol", is_flag=True)
@click.option("--storage-file-path", type=str, default=None)
@click.option("--param-file-path", type=str, default=None)
@click.option("-k", "--model-key", type=str, multiple=True)
@click.option("-i", "--model-id", type=int, multiple=True)
@click.option("--export", is_flag=True)
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--label", type=str, default=None)
@click.option("--export-file-path", type=str, default=None)
@click.option("--disp", is_flag=True)
@click.option("--check-config", is_flag=True)
@click.option("--interactive", is_flag=True)
@click.option("--plot", is_flag=True)
@click.option("--framework", type=str, default='serial')
@click.pass_context
def main(cli, config_file_path, sobol, storage_file_path, param_file_path, model_key, model_id, export,
         output_dir, label, export_file_path, disp, check_config, interactive, plot, framework):
    """
    :param cli: :class:'click.Context': used to process/pass through unknown click arguments
    :param config_file_path: str (path)
    :param sobol: bool
    :param storage_file_path: str (path)
    :param param_file_path: str (path)
    :param model_key: list of str
    :param model_id: list of int
    :param export: bool
    :param output_dir: str
    :param label: str
    :param export_file_path: str
    :param disp: bool
    :param check_config: bool
    :param interactive: bool
    :param plot: bool
    :param framework: str
    """
    # requires a global variable context: :class:'Context'
    kwargs = get_unknown_click_arg_dict(cli.args)
    context.interface = get_parallel_interface(framework, **kwargs)
    context.interface.start(disp=disp)
    context.interface.ensure_controller()
    try:
        nested_analyze_init_controller_context(context, config_file_path, label, output_dir, disp, export_file_path,
                                               interactive=interactive, **kwargs)
        start_time = time.time()
        context.interface.apply(nested_analyze_init_worker_contexts, context.sources, context.update_context_funcs,
                                context.param_names, context.default_params, context.feature_names,
                                context.objective_names, context.target_val, context.target_range, context.label,
                                context.output_dir, context.disp, **context.kwargs)

        for config_collective_func in context.config_collective_funcs:
            context.interface.collective(config_collective_func)

        if disp:
            print('nested.analyze: worker initialization took %.2f s' % (time.time() - start_time))
        sys.stdout.flush()

        if sobol:
            if storage_file_path is None:
                raise RuntimeError('nested.analyze: a storage_file_path must be provided to perform Sobol parameter'
                                   'sensitivity analysis')
            storage = PopulationStorage(file_path=storage_file_path)
            sobol_analysis(config_file_path, storage)
        else:
            param_arrays, model_labels, export_keys, legend = \
                load_model_params(context.param_names, param_file_path=param_file_path,
                                  storage_file_path=storage_file_path, model_keys=model_key, model_ids=model_id)
            if len(param_arrays) < 1:
                if context.x0_array is None:
                    raise Exception('nested.analyze: parameters must be specified either through a config_file, a '
                                    'param_file, or a storage_file')
                param_arrays = [context.x0_array]
                model_labels = [['x0']]
                export_keys = ['0']
                legend = {'model_labels': model_labels, 'export_keys': export_keys,
                          'source': config_file_path}

            if disp:
                print('nested.analyze: evaluating models with the following labels: %s' % model_labels)
                sys.stdout.flush()
            if check_config:
                for param_array in param_arrays:
                    context.interface.apply(update_source_contexts, param_array)
            else:
                features, objectives = evaluate_population(context, param_arrays, export_keys, export, plot)

                if disp:
                    for i, (params, model_label) in enumerate(zip(param_arrays, model_labels)):
                        print('nested.analyze: results for model with labels: %s ' % model_label)
                        this_param_dict = param_array_to_dict(params, context.param_names)
                        print('params:')
                        print_param_dict_like_yaml(this_param_dict)
                        if not features[i] or not objectives[i]:
                            print('nested.analyze: model with labels: %s failed' % model_label)
                        else:
                            if context.feature_names is None:
                                context.feature_names = sorted(list(features[i].keys()))
                            if context.objective_names is None:
                                context.objective_names = sorted(list(objectives[i].keys()))
                            this_features = {key: features[i][key] for key in context.feature_names}
                            this_objectives = {key: objectives[i][key] for key in context.objective_names}
                            print('features:')
                            print_param_dict_like_yaml(this_features)
                            print('objectives:')
                            print_param_dict_like_yaml(this_objectives)
                    sys.stdout.flush()
                    time.sleep(1.)

                if plot:
                    context.interface.show()

                if export:
                    merge_exported_data(context, export_file_path=context.export_file_path,
                                        output_dir=context.output_dir, legend=legend, verbose=disp)

                for shutdown_func in context.shutdown_worker_funcs:
                    context.interface.apply(shutdown_func)

        sys.stdout.flush()
        time.sleep(1.)

        if interactive:
            context.update(locals())
        else:
            context.interface.stop()

    except Exception as e:
        print('nested.analyze: encountered Exception')
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        time.sleep(1.)
        context.interface.stop()
        raise e


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
