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
from nested.optimize_utils import *
from nested.parallel import *
from nested.optimize import evaluate_population
import click, yaml, h5py
from mpi4py import MPI

try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass


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
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--disp", is_flag=True)
@click.option("--check-config", is_flag=True)
@click.option("--interactive", is_flag=True)
@click.option("--plot", is_flag=True)
@click.pass_context
def main(cli, config_file_path, sobol, storage_file_path, param_file_path, model_key, model_id, export,
         output_dir, export_file_path, label, disp, check_config, interactive, plot):
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
    :param export_file_path: str
    :param label: str
    :param disp: bool
    :param check_config: bool
    :param interactive: bool
    :param plot: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    kwargs = get_unknown_click_arg_dict(cli.args)
    context.interface = get_parallel_interface(source_file=__file__, source_package=__package__, **kwargs)
    context.interface.start(disp=disp)
    context.interface.ensure_controller()
    try:
        init_analyze_controller_context(**kwargs)
        start_time = time.time()
        context.interface.apply(init_worker_contexts, context.sources, context.update_context_funcs,
                                context.param_names, context.default_params, context.feature_names,
                                context.objective_names, context.target_val, context.target_range, context.output_dir,
                                context.disp, optimization_title=context.optimization_title, label=context.label,
                                plot=context.plot, export_file_path=context.export_file_path, **context.kwargs)

        for config_synchronize_func in context.config_synchronize_funcs:
            context.interface.synchronize(config_synchronize_func)

        if disp:
            print('nested.analyze: worker initialization took %.2f s' % (time.time() - start_time))
        sys.stdout.flush()

        if sobol:
            if storage_file_path is None:
                raise RuntimeError("Please specify storage-file-path.")
            print("Sobol: performing sensitivity analysis...")
            sys.stdout.flush()
            storage = PopulationStorage(file_path=storage_file_path)
            sobol_analysis(config_file_path, storage)
        else:
            if len(context.model_key) < 1:
                param_arrays = [context.x0_array]
                model_labels = [['x0']]
                export_keys = ['0']
                legend = {'model_labels': model_labels[0], 'export_keys': export_keys,
                          'source': config_file_path}
            else:
                param_arrays, model_labels, export_keys, legend = \
                    load_model_params(context.param_names, context.objective_names,
                                      param_file_path=context.param_file_path,
                                      storage_file_path=context.storage_file_path, model_key=context.model_key)
            if disp:
                print('nested.analyze: evaluating models with the following labels: %s' % model_labels)
                sys.stdout.flush()
            if check_config:
                for param_array in param_arrays:
                    context.interface.apply(update_source_contexts, param_array)
            else:
                features, objectives = evaluate_population(context, param_arrays, export_keys, context.export)

                if disp:
                    for i, (params, model_label) in enumerate(zip(param_arrays, model_labels)):
                        print('nested.analyze: results for model with labels: %s ' % model_labels)
                        try:
                            this_param_dict = param_array_to_dict(params, context.param_names)
                            this_features = {key: features[i][key] for key in context.feature_names}
                            this_objectives = {key: objectives[i][key] for key in context.objective_names}
                            print('params:')
                            pprint.pprint(this_param_dict)
                            print('features:')
                            pprint.pprint(this_features)
                            print('objectives:')
                            pprint.pprint(this_objectives)
                        except Exception as e:
                            print('nested.analyze: model with labels: %s failed' % model_label)
                            raise e
                    sys.stdout.flush()
                    time.sleep(1.)

                if context.plot:
                    context.interface.apply(plt.show)

                if context.export:
                    merge_exported_data(context, export_file_path=context.export_file_path,
                                        output_dir=context.output_dir, legend=legend, verbose=context.disp)

                for shutdown_func in context.shutdown_worker_funcs:
                    context.interface.apply(shutdown_func)
                
        sys.stdout.flush()
        time.sleep(1.)
        if not context.interactive:
            context.interface.stop()
    except Exception as e:
        print('nested.analyze: encountered Exception')
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        time.sleep(1.)
        context.interface.stop()
        raise e


def load_model_params(param_names, objective_names, param_file_path=None, storage_file_path=None, model_key=None,
                    verbose=False):
    """

    :param param_names: list of str
    :param objective_names: list of str
    :param param_file_path: str (path)
    :param storage_file_path: str (path)
    :param model_key: list of str
    :param verbose: bool
    :return: tuple of lists
    """
    if param_file_path is None and storage_file_path is None:
        raise RuntimeError('nested.analyze: either a param_file_path or a storage_file_path must be provided to '
                           'analyze models specified by model_key')

    requested_model_labels = list(model_key)
    ids_to_labels = defaultdict(list)
    labels_to_ids = dict()
    ids_to_param_arrays = dict()
    ids_to_export_keys = dict()
    param_arrays = []
    model_labels = []
    export_keys = []
    legend = {'model_labels': [], 'export_keys': []}

    if storage_file_path is not None:
        if not os.path.isfile(storage_file_path):
            raise Exception('nested.analyze: invalid storage_file_path: %s' % storage_file_path)
        legend['source'] = storage_file_path
        OptRep = OptimizationReport(file_path=storage_file_path)
        if 'all' in requested_model_labels:
            requested_model_labels = ['best'] + list(objective_names)
        for model_label in requested_model_labels:
            if model_label == 'best':
                indiv = OptRep.survivors[0]
            elif model_label not in objective_names:
                raise Exception('nested.analyze: problem finding model_key: %s in storage_file_path: %s' %
                                (model_label, storage_file_path))
            else:
                indiv = OptRep.specialists[model_label]
            id = indiv.model_id
            x = indiv.x
            ids_to_labels[id].append(model_label)
            labels_to_ids[model_label] = id
            ids_to_param_arrays[id] = x
        for export_key, id in enumerate(ids_to_labels):
            this_export_key = str(export_key)
            export_keys.append(this_export_key)
            param_arrays.append(ids_to_param_arrays[id])
            model_labels.append([model_label for model_label in ids_to_labels[id]])
            for model_label in ids_to_labels[id]:
                legend['model_labels'].append(model_label)
                legend['export_keys'].append(this_export_key)

    elif param_file_path is not None:
        if not os.path.isfile(param_file_path):
            raise Exception('nested.analyze: invalid param_file_path: %s' % param_file_path)
        legend['source'] = param_file_path
        with open(param_file_path, 'r') as param_data:
            param_data_dict = yaml.full_load(param_data)

        for export_key, key in enumerate(requested_model_labels):
            if str(key) in param_data_dict:
                this_param_dict = param_data_dict[str(key)]
            elif str(key).isnumeric() and int(key) in param_data_dict:
                this_param_dict = param_data_dict[int(key)]
            else:
                raise RuntimeError('nested.analyze: problem finding model_key: %s in in param_file_path: %s' %
                                   (key, param_file_path))
            this_param_names = list(this_param_dict.keys())
            uncommon_keys = np.setxor1d(this_param_names, param_names)
            if len(uncommon_keys) > 0:
                raise KeyError('parameter_dict for model_key: %s loaded from param_file_path: %s does not match the '
                               'parameter_names specified in the config_file:\n%s' %
                               (key, param_file_path, str(param_names)))
            this_param_array = param_dict_to_array(this_param_dict, param_names)
            param_arrays.append(this_param_array)
            this_export_key = str(export_key)
            export_keys.append(this_export_key)
            model_labels.append([key])
            legend['model_labels'].append(key)
            legend['export_keys'].append(this_export_key)

    return param_arrays, model_labels, export_keys, legend


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
