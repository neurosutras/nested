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
    init_analyze_controller_context(**kwargs)
    start_time = time.time()
    context.interface.apply(init_worker_contexts, context.sources, context.update_context_funcs, context.param_names,
                            context.default_params, context.feature_names, context.objective_names, context.target_val,
                            context.target_range, context.output_dir, context.disp,
                            optimization_title=context.optimization_title, label=context.label, plot=context.plot,
                            **context.kwargs)

    for config_synchronize_func in context.config_synchronize_funcs:
        context.interface.synchronize(config_synchronize_func)

    if disp:
        print('nested.analyze: worker initialization took %.2f s' % (time.time() - start_time))
    sys.stdout.flush()

    try:
        if check_config:
            context.interface.apply(update_source_contexts, context.x0_array)
        elif sobol:
            if storage_file_path is None:
                raise RuntimeError("Please specify storage-file-path.")
            print("Sobol: performing sensitivity analysis...")
            sys.stdout.flush()
            storage = PopulationStorage(file_path=storage_file_path)
            sobol_analysis(config_file_path, storage)
        else:
            if len(context.model_id) < 1 and len(context.model_key) < 1:
                param_arrays = [context.x0_array]
                model_ids = [0]
                model_labels = [[]]
                meta_dict = dict()
            else:
                param_arrays, model_ids, model_labels, meta_dict = \
                    get_model_group(context.param_names, context.objective_names,
                                    param_file_path=context.param_file_path,
                                    storage_file_path=context.storage_file_path, model_id=context.model_id,
                                    model_key=context.model_key, verbose=context.disp)
            features, objectives = evaluate_population(context, param_arrays, model_ids, context.export)

            if context.plot:
                context.interface.apply(plt.show)

            if context.export:
                merge_exported_data(context, param_arrays, model_ids, model_labels, features, objectives,
                                    export_file_path=context.export_file_path, output_dir=context.output_dir,
                                    verbose=context.disp)
                write_metadata(context.export_file_path, meta_dict)
            for shutdown_func in context.shutdown_worker_funcs:
                context.interface.apply(shutdown_func)

            if disp:
                for i, params in enumerate(param_arrays):
                    this_model_id = model_ids[i]
                    this_model_labels = model_labels[i]
                    print('nested.analyze: model_id: %i; model_labels: %s' % (this_model_id, this_model_labels))
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
                        print('nested.analyze: model_id: %i failed' % this_model_id)
                        raise e
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


def write_metadata(file_path, meta_dict):
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        fil = h5py.File(file_path, 'r+')
        for key, val in meta_dict.items():
            fil.attrs[key] = val
        fil.close()
    

def get_model_group(param_names, objective_names, param_file_path=None, storage_file_path=None, model_id=None,
                    model_key=None, verbose=False):
    """

    :param param_names: list of str
    :param objective_names: list of str
    :param param_file_path: str (path)
    :param storage_file_path: str (path)
    :param model_id: list of int or str
    :param model_key: list of str
    :param verbose: bool
    :return: tuple of lists
    """
    if param_file_path is None and storage_file_path is None:
        raise RuntimeError('nested.analyze: either a param_file_path or a storage_file_path must be provided to '
                           'analyze models specified by model_id or model_key')

    requested_param_arrays = []  # list of array
    requested_model_ids = []  # list of int
    requested_model_labels = []  # list of lists of str

    N_model_key = len(model_key)
    meta_dict = {'enum_model': [], 'Storage': False, 'keys': [], 'keys_mod': []}
    
    if storage_file_path is not None:
        if not os.path.isfile(storage_file_path):
            raise Exception('nested.analyze: invalid storage_file_path: %s' % storage_file_path)
        OptRep = StorageModelReport(file_path=storage_file_path)
        obj_names = OptRep.objective_names
        spe_p0 = OptRep.get_category_att()
        spe_arr = OptRep.get_category_id()
        model_id = np.unique(model_id)

        if N_model_key:
            non_best_keys = obj_names if 'all' in model_key else np.array(np.setdiff1d(model_key, 'best'), dtype='S')
            N_non_best = non_best_keys.size
            best = True if N_model_key == N_non_best + 1 else False
            if 'all' in model_key and 'best' in model_key:
                best = True
            mod_key_arr = \
                np.empty(shape=N_non_best,
                         dtype=np.dtype([('key', obj_names.dtype), ('model_id', 'uint32'), ('pos', 'uint32')]))

            for key_idx, key in enumerate(non_best_keys):
                key_pos = np.where(obj_names==key)[0]
                mod_key_arr[key_idx] = key, spe_arr[key_pos], key_pos 
                meta_dict['keys'].append(key)
                meta_dict['keys_mod'].append(spe_arr[key_pos][0])

            uniq_key_models, uniq_idx, inv_idx = np.unique(mod_key_arr['model_id'], return_index=True,
                                                           return_inverse=True)
            uniq_key_p0 = spe_p0[uniq_idx, :]

            if best:
                best_id, best_p0, _, _ = OptRep.get_best_model()
                bestinkeypos = np.where(uniq_key_models==best_id)[0]
                meta_dict['keys'].append('best'.encode())
                meta_dict['keys_mod'].append(best_id)

            if best and not(len(bestinkeypos)):
                best_p0.shape = (1,) + best_p0.shape 
                all_key_uniq = np.append(uniq_key_models, best_id)
                all_key_p0 = np.append(uniq_key_p0, best_p0, axis=0)
            else:
                all_key_uniq = uniq_key_models
                all_key_p0 = uniq_key_p0

            if all_key_p0.ndim == 1:
                all_key_p0.shape = (1,) + all_key_p0.shape

            uniqrelkeys = np.setdiff1d(model_id, all_key_uniq) 
            model_id_list = []
            model_id_p0 = []
            if uniqrelkeys.size:
                uniqrelinspe = np.intersect1d(uniqrelkeys, spe_arr)
                toretmod = np.setdiff1d(uniqrelkeys, uniqrelinspe)

                if uniqrelinspe.size: 
                    idxbool = np.isin(spe_arr, uniqrelinspe)
                    idx = np.nonzero(idxbool)[0] 
                    inspe_id = spe_arr[idx]
                    inspe_p0 = spe_p0[idx, :]

                    if inspe_p0.ndim == 1:
                        inspe_p0.shape = (1,) + inspe_p0.shape

                if toretmod.size:
                    ret_id = toretmod 
                    ret_p0 = OptRep.get_model_att(ret_id, att='x')

                    if ret_p0.ndim == 1:
                        ret_p0.shape = (1,) + ret_p0.shape

                if uniqrelinspe.size and toretmod.size:
                    all_mod_id = np.append(inspe_id, ret_id)
                    all_mod_p0 = np.append(inspe_p0, ret_p0, axis=0)
                elif not(uniqrelinspe.size):
                    all_mod_id = ret_id
                    all_mod_p0 = ret_p0
                else:
                    all_mod_id = inspe_id
                    all_mod_p0 = inspe_p0

                all_key_mod_id = np.append(all_key_uniq, all_mod_id)
                all_key_mod_p0 = np.append(all_key_p0, all_mod_p0, axis=0)
            else:
                all_key_mod_id = all_key_uniq
                all_key_mod_p0 = all_key_p0
        else:
            all_key_mod_id = model_id 
            all_key_mod_p0 = OptRep.get_model_att(model_id, att='x') 
            
        OptRep.close_file()

        meta_dict['Storage'] = True 

        for enum_idx, (mod_id, p0) in enumerate(zip(all_key_mod_id, all_key_mod_p0)):
            requested_model_labels.append('{:d}'.format(mod_id))
            requested_param_arrays.append(tuple(p0))
            requested_model_ids.append(enum_idx)
            meta_dict['enum_model'].append(mod_id)

    elif param_file_path is not None:
        if not os.path.isfile(param_file_path):
            raise Exception('nested.analyze: invalid param_file_path: %s' % param_file_path)
      
        with open(param_file_path, 'r') as param_data:
            param_data_dict = yaml.full_load(param_data)

        param_dd_keys = list(param_data_dict.keys())
        param_dd_params = list(param_data_dict[param_dd_keys[0]].keys())

        uncommon_keys = np.setxor1d(param_dd_params, param_names)
    
        if uncommon_keys.size:
            raise KeyError('Optimization parameter mismatch between config and param files: {!s}'.format(uncommon_keys))
    
        meta_dict['Storage'] = False 

        for kidx, key in enumerate(model_key): 
            requested_model_labels.append('{:d}'.format(kidx))
            if str(key) in param_data_dict:
                requested_param_arrays.append(param_dict_to_array(param_data_dict[str(key)], param_names))
            elif str(key).isnumeric() and int(key) in param_data_dict:
                requested_param_arrays.append(param_dict_to_array(param_data_dict[int(key)], param_names))
            else:
                raise RuntimeError('nested.analyze: provided model_key: %s not found in param_file_path: %s' %
                                   (key, param_file_path))
            requested_model_ids.append(kidx)
            meta_dict['enum_model'].append(kidx)
            meta_dict['keys'].append(key.encode())
            meta_dict['keys_mod'].append(kidx)

    return requested_param_arrays, requested_model_ids, requested_model_labels, meta_dict

#    if param_file_path is not None:
#        if not os.path.isfile(param_file_path):
#            raise Exception('nested.analyze: invalid param_file_path: %s' % param_file_path)
#        if model_key is None or len(model_key) < 1:
#            raise RuntimeError('nested.analyze: missing required parameter: a model_key must be provided to to analyze '
#                               'models specified by a param_file_path: %s' % param_file_path)
#        model_param_dict = read_from_yaml(param_file_path)
#        if model_id is None or len(model_id) < 1:
#            requested_model_ids = list(range(len(model_key)))
#        elif len(model_id) != len(model_key):
#            raise RuntimeError('nested.analyze: when providing both model_keys for import and model_ids for export, '
#                               'they must be the same length')
#        else:
#            requested_model_ids = list(model_id)
#        for this_model_key in model_key:
#            if str(this_model_key) in model_param_dict:
#                requested_param_arrays.append(param_dict_to_array(model_param_dict[str(this_model_key)], param_names))
#                requested_model_labels.append([str(this_model_key)])
#            elif str(this_model_key).isnumeric() and int(this_model_key) in model_param_dict:
#                requested_param_arrays.append(param_dict_to_array(model_param_dict[int(this_model_key)], param_names))
#                requested_model_labels.append([str(this_model_key)])
#            else:
#                raise RuntimeError('nested.analyze: provided model_key: %s not found in param_file_path: %s' %
#                                   (str(this_model_key), param_file_path))

#        if model_id is not None and len(model_id) > 0:
#            with h5py.File(context.storage_file_path, 'r') as f:
#                count = 0
#                for group in f.values():
#                    if 'count' in group.attrs:
#                        count = max(count, group.attrs['count'])
#            for this_model_id in model_id:
#                if int(this_model_id) >= count:
#                    raise RuntimeError('nested.analyze: invalid model_id: %i' % int(this_model_id))
#                requested_model_ids.append(int(this_model_id))
#                # TODO: find requested param_arrays and any associated labels in file
#
#
#        else:
#            if model_key is None or len(model_key) < 1:
#                model_key = ('best',)
#                if verbose:
#                    print('nested.analyze: no model_id or model_key specified; defaulting to evaluate best model in '
#                          'storage_file_path: %s' % storage_file_path)
#                    sys.stdout.flush()
#            valid_model_keys = set(objective_names)
#            valid_model_keys.add('best')
#            for i, this_model_key in enumerate(model_key):
#                if str(this_model_key) not in valid_model_keys:
#                    raise RuntimeError('nested.analyze: invalid model_key: %s' % str(this_model_key))
#                if this_model_key == 'best':
#                    report = OptimizationReport(file_path=context.storage_file_path)
#                    requested_param_arrays.append(report.survivors[0].x)
#                    requested_model_ids.append(i)
#                    # TODO: need to also append any additional (specialist) labels
#                    requested_model_labels.append([str(this_model_key)])
#                else:
#                    # TODO: find requested param_arrays, model_ids and any additional associated labels in file
#                    pass
#


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
