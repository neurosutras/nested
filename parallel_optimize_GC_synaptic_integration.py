__author__ = 'Grace Ng and Aaron D. Milstein'
from specify_cells4 import *
from plot_results import *
from moopgen import *

"""
Submodule used by parallel_optimize to tune properties of synaptic AMPA- and NMDA-type glutamate receptors in dentate
granule cells to match target features of spatiotemporal integration.
Requires a YAML file to specify required configuration parameters. 
Requires use of an ipyparallel client.
"""

context = Context()


def config_interactive(config_file_path='data/parallel_optimize_GC_synaptic_integration_config.yaml',
                       output_dir='data', temp_output_path=None, export_file_path=None, verbose=True, disp=True):
    """
    :param config_file_path: str (.yaml file path)
    :param output_dir: str (dir path)
    :param temp_output_path: str (.hdf5 file path)
    :param export_file_path: str (.hdf5 file path)
    :param verbose: bool
    :param disp: bool
    """
    config_dict = read_from_yaml(config_file_path)
    if 'param_gen' in config_dict and config_dict['param_gen'] is not None:
        param_gen_name = config_dict['param_gen']
    else:
        param_gen_name = 'BGen'
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

    if 'update_params' not in config_dict or config_dict['update_params'] is None:
        update_params = []
    else:
        update_params = config_dict['update_params']
    update_params_funcs = []
    for update_params_func_name in update_params:
        func = globals().get(update_params_func_name)
        if not callable(func):
            raise Exception('parallel_optimize: update_params: %s is not a callable function.'
                            % update_params_func_name)
        update_params_funcs.append(func)

    if temp_output_path is None:
        temp_output_path = '%s/parallel_optimize_temp_output_%s_pid%i.hdf5' % \
                        (output_dir, datetime.datetime.today().strftime('%m%d%Y%H%M'), os.getpid())
    if export_file_path is None:
        export_file_path = '%s/%s_%s_%s_optimization_exported_output.hdf5' % \
                           (output_dir, datetime.datetime.today().strftime('%m%d%Y%H%M'), optimization_title,
                            param_gen_name)
    x0_array = param_dict_to_array(x0, param_names)
    context.update(locals())
    context.update(kwargs)
    config_engine(update_params_funcs, param_names, default_params, temp_output_path, export_file_path, output_dir, disp,
                  **kwargs)
    update_submodule_params(x0_array)


def config_controller(export_file_path, **kwargs):
    """

    :param export_file_path: str
    """
    processed_export_file_path = export_file_path.replace('.hdf5', '_processed.hdf5')
    context.update(locals())
    context.update(kwargs)
    init_context()


def config_engine(update_params_funcs, param_names, default_params, temp_output_path, export_file_path, output_dur, disp,
                  mech_file_path, neuroH5_file_path, neuroH5_index, spines, **kwargs):
    """
    :param update_params_funcs: list of function references
    :param param_names: list of str
    :param default_params: dict
    :param temp_output_path: str
    :param export_file_path: str
    :param output_dur: str (dir path)
    :param disp: bool
    :param mech_file_path: str
    :param neuroH5_file_path: str
    :param neuroH5_index: int
    :param spines: bool
    """
    neuroH5_dict = read_from_pkl(neuroH5_file_path)[neuroH5_index]
    param_indexes = {param_name: i for i, param_name in enumerate(param_names)}
    processed_export_file_path = export_file_path.replace('.hdf5', '_processed.hdf5')
    context.update(locals())
    context.update(kwargs)
    init_context()
    setup_cell(**kwargs)


def init_context():
    """

    :return:
    """
    seed_offset = 7. * 2e6
    # for clustered inputs, num_syns corresponds to number of clustered inputs per branch
    num_syns = {'random': 30, 'clustered': 20}  # {'random': 30, 'clustered': 20}
    # num_syns = {'random': 5, 'clustered': 5}  # {'random': 30, 'clustered': 20}
    # number of branches to test temporal integration of clustered inputs
    num_clustered_branches = 2
    clustered_branch_names = ['clustered%i' % i for i in xrange(num_clustered_branches)]
    synapse_indexes = {'random': range(num_syns['random'])}
    for i, branch in enumerate(clustered_branch_names):
        synapse_indexes[branch] = range(num_syns['random'] + i * num_syns['clustered'],
                                        num_syns['random'] + (i + 1) * num_syns['clustered'])
    syn_conditions = ['control', 'AP5', 'TTX']
    ISI = {'units': 150., 'clustered': 1.1}  # inter-stimulus interval for synaptic stim (ms)
    units_per_sim = 5
    equilibrate = 250.  # time to steady-state
    stim_dur = 150.
    sim_duration = {'units': equilibrate + units_per_sim * ISI['units'],
                    'clustered': equilibrate + 200.,
                    'default': equilibrate + stim_dur}
    trace_baseline = 10.
    duration = max(sim_duration.values())
    th_dvdt = 10.
    dt = 0.02
    v_init = -77.
    v_active = -77.
    NMDA_type = 'NMDA_KIN5'
    syn_types = ['AMPA_KIN', NMDA_type]
    local_random = random.Random()
    i_holding = {'soma': 0.}
    i_th = {'soma': 0.1}
    context.update(locals())


def setup_cell(verbose=False, cvode=False, daspk=False, **kwargs):
    """

    :param verbose: bool
    :param cvode: bool
    :param daspk: bool
    """
    cell = DG_GC(neuroH5_dict=context.neuroH5_dict, mech_file_path=context.mech_file_path,
                 full_spines=context.spines)
    context.cell = cell
    context.local_random.seed(int(context.seed_offset + context.neuroH5_index))

    syn_list = []

    # Choose random synapses, which will be used to calculate the average unitary EPSP
    syn_pointer_list = []  # list of tuple : (branch.index, syn_id, exc_syn_index)
    this_syn_pointer_list = []
    for branch in cell.apical:
        this_synapse_attributes = branch.get_filtered_synapse_attributes(syn_category='excitatory')
        if len(this_synapse_attributes['syn_locs']) > 1:
            candidates = [(branch.index, syn_id, exc_syn_index)
                          for (exc_syn_index, syn_id) in enumerate(this_synapse_attributes['syn_id'])]
            if branch.sec.L <= 10:
                this_syn_pointer_list.extend(context.local_random.sample(candidates, 1))
            else:
                this_num_syns = min(len(this_synapse_attributes['syn_locs']), int(branch.sec.L / 10.))
                this_syn_pointer_list.extend(context.local_random.sample(candidates, this_num_syns))
        elif this_synapse_attributes['syn_locs']:
            this_syn_pointer_list.append(this_synapse_attributes['syn_id'][0])
    syn_pointer_list.extend(context.local_random.sample(this_syn_pointer_list, context.num_syns['random']))

    # Choose synapses from apical branches to measure spatiotemporal integration of clustered inputs.
    # Each branch must have > 30 synapses within 30 um, choose synapses near the middle of the branch.
    candidate_branches = [apical for apical in cell.apical if
                          50. < cell.get_distance_to_node(cell.tree.root, apical) < 150. and
                          apical.sec.L > 80.]
    context.local_random.shuffle(candidate_branches)

    parents = []
    success_branches = 0
    for branch in (branch for branch in candidate_branches if branch.parent not in parents):
        this_synapse_attributes = branch.get_filtered_synapse_attributes(syn_category='excitatory')
        if this_synapse_attributes['syn_locs']:
            candidates = [(branch.index, syn_id, exc_syn_index)
                          for (exc_syn_index, syn_id) in enumerate(this_synapse_attributes['syn_id'])
                          if 30. <= this_synapse_attributes['syn_locs'][exc_syn_index] * branch.sec.L <= 60.]
            if len(candidates) >= context.num_syns['clustered']:
                syn_pointer_list.extend(context.local_random.sample(candidates, context.num_syns['clustered']))
                success_branches += 1
                parents.append(branch.parent)
        if success_branches > 1:
            break
    if success_branches < 2:
        raise Exception('Only %i/%i branches satisfy the requirements for clustered synaptic input.' %
                        (success_branches, context.num_clustered_branches))

    for branch_index, syn_id, exc_syn_index in syn_pointer_list:
        branch = context.cell.tree.get_node_with_index(branch_index)
        if context.spines:
            spine = branch.spines[exc_syn_index]
            syn = Synapse(context.cell, spine, syn_types=context.syn_types, stochastic=False, loc=0.5, id=syn_id)
        else:
            syn = Synapse(context.cell, branch, syn_types=context.syn_types, stochastic=False, id=syn_id)
        syn_list.append(syn)
    context.syn_list = syn_list

    # get the thickest apical dendrite ~200 um from the soma
    candidate_branches = []
    candidate_diams = []
    candidate_locs = []
    for branch in cell.apical:
        if ((cell.get_distance_to_node(cell.tree.root, branch, 0.) >= 200.) &
                (cell.get_distance_to_node(cell.tree.root, branch, 1.) > 300.) & (not cell.is_terminal(branch))):
            candidate_branches.append(branch)
            for seg in branch.sec:
                loc = seg.x
                if cell.get_distance_to_node(cell.tree.root, branch, loc) > 250.:
                    candidate_diams.append(branch.sec(loc).diam)
                    candidate_locs.append(loc)
                    break
    index = candidate_diams.index(max(candidate_diams))
    dend = candidate_branches[index]
    dend_loc = candidate_locs[index]
    rec_locs = {'soma': 0., 'dend': dend_loc, 'local_branch': 0.5}
    context.rec_locs = rec_locs
    rec_nodes = {'soma': cell.tree.root, 'dend': dend, 'local_branch': dend}
    context.rec_nodes = rec_nodes

    equilibrate = context.equilibrate
    duration = context.duration
    dt = context.dt
    stim_dur = context.stim_dur

    sim = QuickSim(duration, cvode=cvode, daspk=daspk, dt=dt, verbose=verbose)
    sim.parameters['duration'] = duration
    sim.parameters['equilibrate'] = equilibrate
    sim.parameters['spines'] = context.spines
    sim.append_stim(cell, cell.tree.root, loc=0., amp=0., delay=equilibrate, dur=stim_dur, description='step')
    sim.append_stim(cell, cell.tree.root, loc=0., amp=0., delay=0., dur=duration, description='offset')
    for description, node in rec_nodes.iteritems():
        sim.append_rec(cell, node, loc=rec_locs[description], description=description)
    context.sim = sim

    context.spike_output_vec = h.Vector()
    cell.spike_detector.record(context.spike_output_vec)


def update_submodule_params(x, local_context=None):
    """

    :param x: array
    :param local_context: :class:'Context'
    """
    if local_context is None:
        local_context = context
    local_context.cell.reinit_mechanisms(from_file=True)
    if not local_context.spines:
        local_context.cell.correct_g_pas_for_spines()
    for update_func in local_context.update_params_funcs:
        update_func(x, local_context)


def get_unitary_EPSP_features(indiv, c=None, client_range=None, export=False):
    """
    Distribute simulations across available engines for measuring unitary EPSP amplitude.
    :param indiv: dict {'pop_id': pop_id, 'x': x arr, 'features': features dict}
    :param c: :class:'ipyparallel.Client'
    :param client_range: list of int
    :param export: bool
    :return: dict
    """
    if c is not None:
        if client_range is None:
            client_range = range(len(c))
        dv = c[client_range]
        map_func = dv.map_async
    else:
        map_func = map
    x = indiv['x']
    syn_group_list = []
    syn_index_list = []
    syn_condition_list = []
    for syn_group in context.synapse_indexes:
        this_syn_index_list = []
        start = 0
        while start < len(context.synapse_indexes[syn_group]):
            this_syn_index_list.append(context.synapse_indexes[syn_group][start:start+context.units_per_sim])
            start += context.units_per_sim
        num_sims = len(this_syn_index_list)
        syn_index_list.extend(this_syn_index_list)
        syn_group_list.extend([syn_group] * num_sims)
        syn_condition_list.extend(['control'] * num_sims)
        if syn_group == 'random':
            syn_index_list.extend(this_syn_index_list)
            syn_group_list.extend([syn_group] * num_sims)
            syn_condition_list.extend(['AP5'] * num_sims)
    result = map_func(compute_EPSP_amp_features, [x] * len(syn_index_list), syn_index_list, syn_condition_list,
                      syn_group_list, [export] * len(syn_index_list))
    return {'pop_id': indiv['pop_id'], 'client_range': client_range, 'async_result': result,
            'filter_features': filter_unitary_EPSP_features}


def compute_EPSP_amp_features(x, syn_indexes, syn_condition, syn_group, export=False, plot=False):
    """

    :param x: arr
    :param syn_indexes: list of int
    :param syn_condition: str
    :param syn_group: str
    :param export: bool
    :param plot: bool
    :return: dict
    """
    start_time = time.time()
    update_submodule_params(x, context)

    soma_vm = offset_vm('soma', context.v_init)
    synapses = [context.syn_list[syn_index] for syn_index in syn_indexes]
    for i, syn in enumerate(synapses):
        syn.source.play(h.Vector([context.equilibrate + i * context.ISI['units']]))
        if i == 0:
            context.sim.parameters['input_loc'] = syn.branch.type
            context.sim.modify_rec(context.sim.get_rec_index('local_branch'), node=syn.branch)
        if syn_condition == 'AP5':
            syn.target(context.NMDA_type).gmax = 0.
    description = 'condition: %s, group: %s, syn_indexes: %i:%i' % (syn_condition, syn_group, syn_indexes[0],
                                                                    syn_indexes[-1])
    title = 'EPSP_amp_features'
    duration = context.sim_duration['units']
    context.sim.tstop = duration
    context.sim.parameters['duration'] = duration
    context.sim.parameters['description'] = description
    context.sim.parameters['title'] = title
    context.sim.run(context.v_init)
    dt = context.dt
    equilibrate = context.equilibrate
    interp_t = np.arange(0., duration, dt)
    trace_baseline = context.trace_baseline

    result = {}
    for i, syn in enumerate(synapses):
        start = int((equilibrate + i * context.ISI['units']) / dt)
        end = start + int(context.ISI['units'] / dt)
        trace_start = start - int(trace_baseline / dt)
        baseline_start, baseline_end = int(start - 3. / dt), int(start - 1. / dt)
        syn_id = syn_indexes[i]
        if syn_id not in result:
            result[syn_id] = {'traces': {}}
        for rec in context.sim.rec_list:
            interp_vm = np.interp(interp_t, context.sim.tvec, rec['vec'])
            baseline = np.mean(interp_vm[baseline_start:baseline_end])
            corrected_vm = interp_vm[trace_start:end] - baseline
            peak = np.max(corrected_vm)
            peak_index = np.where(corrected_vm == peak)[0][0]
            zero_index = np.where(corrected_vm[peak_index:] <= 0.)[0]
            if np.any(zero_index):
                corrected_vm[peak_index+zero_index[0]:] = 0.
            if rec['description'] == 'soma':
                result[syn_id]['soma_EPSP_amp'] = peak
            result[syn_id]['traces'][rec['description']] = np.array(corrected_vm)
        syn.source.play(h.Vector())

    result = {syn_group: {syn_condition: result}}
    if context.disp:
        print 'Process: %i: %s: %s took %.3f s' % (os.getpid(), title, description, time.time() - start_time)
    if plot:
        context.sim.plot()
    if export:
        export_sim_results()
    return result


def filter_unitary_EPSP_features(computed_result_list, current_features, target_val, target_range, export=False):
    """

    :param computed_result_list: list of dict (each dict contains results from a single simulation)
    :param current_features: dict
    :param target_val: dict of float
    :param target_range: dict of float
    :param export: bool
    :return: dict
    """
    soma_EPSP_amp_dict = {}
    traces = {}
    for this_result_dict in computed_result_list:
        for syn_group in this_result_dict:
            if syn_group not in traces:
                traces[syn_group] = {}
            for syn_condition in this_result_dict[syn_group]:
                if syn_condition not in traces[syn_group]:
                    traces[syn_group][syn_condition] = {}
                if syn_group == 'random' and syn_condition not in soma_EPSP_amp_dict:
                    soma_EPSP_amp_dict[syn_condition] = {}
                for syn_id in this_result_dict[syn_group][syn_condition]:
                    if syn_group == 'random':
                        soma_EPSP_amp_dict[syn_condition][syn_id] = \
                            this_result_dict[syn_group][syn_condition][syn_id]['soma_EPSP_amp']
                    traces[syn_group][syn_condition][syn_id] = \
                        this_result_dict[syn_group][syn_condition][syn_id]['traces']
    NMDA_contribution_list = [(soma_EPSP_amp_dict['control'][syn_id] - soma_EPSP_amp_dict['AP5'][syn_id]) /
                              soma_EPSP_amp_dict['control'][syn_id] for syn_id in soma_EPSP_amp_dict['control']]
    new_features = {'soma_EPSP_amp': np.mean(soma_EPSP_amp_dict['control'].values()),
                    'NMDA_contribution': np.mean(NMDA_contribution_list),
                    'unitary_EPSP_traces': traces}
    if export:
        t = np.arange(-context.trace_baseline, context.ISI['units'], context.dt)
        rec_names = traces['random']['control'].itervalues().next().keys()
        description = 'unitary_EPSP_features'
        with h5py.File(context.processed_export_file_path, 'a') as f:
            if description not in f:
                f.create_group(description)
            if 'time' not in f[description]:
                f[description].create_dataset('time', compression='gzip', compression_opts=9, data=t)
            if 'data' not in f[description]:
                f[description].create_group('data')
            for syn_condition in traces['random']:
                if syn_condition not in f[description]['data']:
                    f[description]['data'].create_group(syn_condition)
                f[description]['data'][syn_condition].create_group('mean_traces')
                for rec in rec_names:
                    this_mean_trace = np.mean([traces['random'][syn_condition][syn_id][rec]
                                               for syn_id in traces['random'][syn_condition]], axis=0)
                    f[description]['data'][syn_condition]['mean_traces'].create_dataset(rec, compression='gzip',
                                                                                        compression_opts=9,
                                                                                        data=this_mean_trace)
    return new_features


def get_compound_EPSP_features(indiv, c=None, client_range=None, export=False):
    """
    Distribute simulations across available engines for measuring compound EPSP amplitude.
    :param indiv: dict {'pop_id': pop_id, 'x': x arr, 'features': features dict}
    :param c: :class:'ipyparallel.Client'
    :param client_range: list of int
    :param export: bool
    :return: dict
    """
    if c is not None:
        if client_range is None:
            client_range = range(len(c))
        dv = c[client_range]
        map_func = dv.map_async
    else:
        map_func = map
    x = indiv['x']
    syn_group_list = []
    syn_index_list = []
    syn_condition_list = []
    for syn_group in context.clustered_branch_names:
        this_syn_index_list = [context.synapse_indexes[syn_group][:i + 1] for i in
                               range(len(context.synapse_indexes[syn_group]))]
        num_sims = len(this_syn_index_list)
        for syn_condition in context.syn_conditions:
            syn_index_list.extend(this_syn_index_list)
            syn_group_list.extend([syn_group] * num_sims)
            syn_condition_list.extend([syn_condition] * num_sims)
    result = map_func(compute_branch_cooperativity_features, [x] * len(syn_index_list), syn_index_list,
                      syn_condition_list, syn_group_list, [export] * len(syn_index_list))
    return {'pop_id': indiv['pop_id'], 'client_range': client_range, 'async_result': result,
            'filter_features': filter_compound_EPSP_features}


def compute_branch_cooperativity_features(x, syn_indexes, syn_condition, syn_group, export=False, plot=False):
    """
    The specified synapses are stimulated quasi-synchronously to measure synaptic cooperativity (spatiotemporal
    integration).
    :param x: arr
    :param syn_indexes: list of int
    :param syn_condition: str
    :param syn_group: str
    :param export: bool
    :param plot: bool
    :return: dict
    """
    start_time = time.time()
    update_submodule_params(x, context)
    if syn_condition == 'TTX':
        context.cell.zero_na()

    soma_vm = offset_vm('soma', context.v_init)
    synapses = [context.syn_list[syn_index] for syn_index in syn_indexes]
    for i, syn in enumerate(synapses):
        syn.source.play(h.Vector([context.equilibrate + i * context.ISI['clustered']]))
        if i == 0:
            context.sim.parameters['input_loc'] = syn.branch.type
            context.sim.modify_rec(context.sim.get_rec_index('local_branch'), node=syn.branch)
        if syn_condition == 'AP5':
            syn.target(context.NMDA_type).gmax = 0.
    num_syns = len(syn_indexes)
    description = 'condition: %s, group: %s, num_syns: %i' % (syn_condition, syn_group, num_syns)
    title = 'branch_cooperativity_features'
    duration = context.sim_duration['clustered']
    context.sim.tstop = duration
    context.sim.parameters['duration'] = duration
    context.sim.parameters['description'] = description
    context.sim.parameters['title'] = title
    context.sim.run(context.v_init)
    dt = context.dt
    equilibrate = context.equilibrate
    interp_t = np.arange(0., duration, dt)
    trace_baseline = context.trace_baseline
    start = int(equilibrate / dt)
    trace_start = start - int(trace_baseline / dt)
    baseline_start, baseline_end = int(start - 3. / dt), int(start - 1. / dt)

    result = {'syn_indexes': syn_indexes, 'traces': {}}
    for rec in context.sim.rec_list:
        interp_vm = np.interp(interp_t, context.sim.tvec, rec['vec'])
        baseline = np.mean(interp_vm[baseline_start:baseline_end])
        corrected_vm = interp_vm[trace_start:] - baseline
        result['traces'][rec['description']] = np.array(corrected_vm)
    for i, syn in enumerate(synapses):
        syn.source.play(h.Vector())

    result = {syn_group: {syn_condition: {num_syns: result}}}
    if context.disp:
        print 'Process: %i: %s: %s took %.3f s' % (os.getpid(), title, description, time.time() - start_time)
    if plot:
        context.sim.plot()
    if export:
        export_sim_results()
    return result


def filter_compound_EPSP_features(computed_result_list, current_features, target_val, target_range, export=False):
    """

    :param computed_result_list: list of dict (each dict contains results from a single simulation)
    :param current_features: dict
    :param target_val: dict of float
    :param target_range: dict of float
    :param export: bool
    :return: dict
    """
    soma_EPSP_amp_dict = {}
    traces = {}
    syn_indexes = {}
    for this_result_dict in computed_result_list:
        for syn_group in this_result_dict:
            if syn_group not in traces:
                traces[syn_group] = {}
                syn_indexes[syn_group] = {}
            for syn_condition in this_result_dict[syn_group]:
                if syn_condition not in traces[syn_group]:
                    traces[syn_group][syn_condition] = {}
                for num_syns in this_result_dict[syn_group][syn_condition]:
                    traces[syn_group][syn_condition][num_syns] = \
                        this_result_dict[syn_group][syn_condition][num_syns]['traces']
                    if syn_condition == 'control':
                        syn_indexes[syn_group][num_syns] = \
                            this_result_dict[syn_group][syn_condition][num_syns]['syn_indexes']
    expected_traces = get_expected_compound_EPSP_traces(current_features, syn_indexes)
    # only need expected for syn_condition == 'control'
    for syn_group in expected_traces:
        traces[syn_group]['expected'] = expected_traces[syn_group]
    t = np.arange(-context.trace_baseline, context.sim_duration['clustered'] - context.equilibrate, context.dt)
    for syn_group in traces:
        if syn_group not in soma_EPSP_amp_dict:
            soma_EPSP_amp_dict[syn_group] = {}
        for syn_condition in traces[syn_group]:
            if syn_condition not in soma_EPSP_amp_dict[syn_group]:
                soma_EPSP_amp_dict[syn_group][syn_condition] = {}
            for num_syns in traces[syn_group][syn_condition]:
                if syn_condition == 'expected':
                    soma_EPSP_amp_dict[syn_group][syn_condition][num_syns] = \
                        np.max(traces[syn_group][syn_condition][num_syns]['soma'])
                else:
                    spikes_removed = remove_spikes_from_array(traces[syn_group][syn_condition][num_syns]['soma'], t,
                                                              dt=context.dt)
                    soma_EPSP_amp_dict[syn_group][syn_condition][num_syns] = np.max(spikes_removed)
    integration_gain = {}
    initial_gain = {}
    soma_EPSP_amp_array = {}
    for syn_group in soma_EPSP_amp_dict:
        soma_EPSP_amp_array[syn_group] = {}
        num_syns_list = np.arange(1, len(soma_EPSP_amp_dict[syn_group]['control']) + 1)
        this_expected = np.array([soma_EPSP_amp_dict[syn_group]['expected'][num_syns] for num_syns in num_syns_list])
        soma_EPSP_amp_array[syn_group]['expected'] = this_expected
        for syn_condition in context.syn_conditions:
            if syn_condition not in integration_gain:
                integration_gain[syn_condition] = {}
                initial_gain[syn_condition] = {}
            this_actual = np.array([soma_EPSP_amp_dict[syn_group][syn_condition][num_syns]
                                    for num_syns in num_syns_list])
            soma_EPSP_amp_array[syn_group][syn_condition] = this_actual
            this_ratio = np.divide(this_actual, this_expected)
            # Integration should be close to linear without gain for the first few synapses.
            initial_gain[syn_condition][syn_group] = np.mean(this_ratio[:2])
            slope, intercept, r_value, p_value, std_err = stats.linregress(this_expected, this_actual)
            integration_gain[syn_condition][syn_group] = slope
    new_features = {}
    for syn_condition in context.syn_conditions:
        new_features['initial_gain_' + syn_condition] = np.mean(initial_gain[syn_condition].values())
        new_features['integration_gain_'+syn_condition] = np.mean(integration_gain[syn_condition].values())
    if export:
        description = 'compound_EPSP_features'
        with h5py.File(context.processed_export_file_path, 'a') as f:
            if description not in f:
                f.create_group(description)
            if 'time' not in f[description]:
                f[description].create_dataset('time', compression='gzip', compression_opts=9, data=t)
            if 'data' not in f[description]:
                f[description].create_group('data')
            for syn_group in traces:
                if syn_group not in f[description]['data']:
                    f[description]['data'].create_group(syn_group)
                for syn_condition in traces[syn_group]:
                    if syn_condition not in f[description]['data'][syn_group]:
                        f[description]['data'][syn_group].create_group(syn_condition)
                    if 'traces' not in f[description]['data'][syn_group][syn_condition]:
                        f[description]['data'][syn_group][syn_condition].create_group('traces')
                    for num_syns in traces[syn_group][syn_condition]:
                        num_syns_key = str(num_syns)
                        if num_syns_key not in f[description]['data'][syn_group][syn_condition]['traces']:
                            num_syns_group = f[description]['data'][syn_group][syn_condition]['traces'].create_group(
                                num_syns_key)
                        for rec in traces[syn_group][syn_condition][num_syns]:
                            num_syns_group.create_dataset(rec, compression='gzip', compression_opts=9,
                                                          data=traces[syn_group][syn_condition][num_syns][rec])
                    f[description]['data'][syn_group][syn_condition].create_dataset(
                        'EPSP_amp', compression='gzip', compression_opts=9,
                        data=soma_EPSP_amp_array[syn_group][syn_condition])
    return new_features


def get_spike_shape_features(indiv, c, client_range, export=False):
    """
    Distribute simulations across available engines for testing spike stability.
    :param indiv: dict {'pop_id': pop_id, 'x': x arr, 'features': features dict}
    :param c: Client object
    :param client_range: list of ints
    :param export: False (for exporting voltage traces)
    :return: dict
    """
    dv = c[client_range]
    x = indiv['x']
    result = dv.map_async(compute_spike_shape_features, [x], [export])
    return {'pop_id': indiv['pop_id'], 'client_range': client_range, 'async_result': result}


def compute_spike_shape_features(x, export=False, plot=False):
    """
    :param x: array
    :param export: bool
    :param plot: bool
    :return: dict
    """
    start_time = time.time()
    update_submodule_params(x, context)

    v_active = context.v_active
    equilibrate = context.equilibrate
    dt = context.dt
    i_th = context.i_th

    soma_vm = offset_vm('soma', v_active)
    stim_dur = 150.
    step_stim_index = context.sim.get_stim_index('step')
    context.sim.modify_stim(step_stim_index, node=context.cell.tree.root, loc=0., dur=stim_dur)
    duration = equilibrate + stim_dur
    context.sim.tstop = duration
    t = np.arange(0., duration, dt)
    spike = False
    d_amp = 0.01
    amp = max(0., i_th['soma'] - 0.02)
    while not spike:
        context.sim.modify_stim(step_stim_index, amp=amp)
        context.sim.run(v_active)
        vm = np.interp(t, context.sim.tvec, context.sim.get_rec('soma')['vec'])
        if np.any(vm[:int(equilibrate/dt)] > -30.):
            print 'Process %i: Aborting - spontaneous firing' % (os.getpid())
            return None
        if np.any(vm[int(equilibrate/dt):int((equilibrate+50.)/dt)] > -30.):
            spike = True
        elif amp >= 0.4:
            print 'Process %i: Aborting - rheobase outside target range' % (os.getpid())
            return None
        else:
            amp += d_amp
            if context.sim.verbose:
                print 'increasing amp to %.3f' % amp
    title = 'spike_shape_features'
    description = 'step current amp: %.3f' % amp
    context.sim.parameters['description'] = description
    context.sim.parameters['title'] = title
    context.sim.parameters['duration'] = duration
    i_th['soma'] = amp
    spike_times = context.cell.spike_detector.get_recordvec().to_python()
    peak, threshold, ADP, AHP = get_spike_shape(vm, spike_times)
    dend_vm = np.interp(t, context.sim.tvec, context.sim.get_rec('dend')['vec'])
    th_x = np.where(vm[int(equilibrate / dt):] >= threshold)[0][0] + int(equilibrate / dt)
    if len(spike_times) > 1:
        end = min(th_x + int(10. / dt), int((spike_times[1] - 5.)/dt))
    else:
        end = th_x + int(10. / dt)
    dend_peak = np.max(dend_vm[th_x:end])
    dend_pre = np.mean(dend_vm[int((equilibrate - 3.) / dt):int((equilibrate - 1.) / dt)])
    result = {'dend_AP_amp': (dend_peak - dend_pre) / (peak - soma_vm)}
    print 'Process %i: %s: %s took %.1f s' % (os.getpid(), title, description, time.time() - start_time)
    if plot:
        context.sim.plot()
    if export:
        export_sim_results()
    return result


def get_objectives(features, target_val, target_range):
    """

    :param features: dict
    :param target_val: dict of float
    :param target_range: dict of float
    :return: tuple of dict
    """
    objectives = {}
    objective_names = ['dend_AP_amp', 'soma_EPSP_amp', 'NMDA_contribution', 'integration_gain_control',
                       'integration_gain_TTX', 'integration_gain_AP5', 'initial_gain_control', 'initial_gain_TTX',
                       'initial_gain_AP5']
    for objective_name in objective_names:
        objectives[objective_name] = ((target_val[objective_name] - features[objective_name]) /
                                      target_range[objective_name]) ** 2.
    return features, objectives


def get_expected_compound_EPSP_traces(current_features, syn_indexes, syn_condition=None):
    """

    :param current_features: dict
    :param syn_indexes: dict
    :param syn_condition: str
    :param rec_description: str
    :return:
    """
    if syn_condition is None:
        syn_condition = 'control'
    traces = {}
    baseline_len = int(context.trace_baseline / context.dt)
    unitary_len = int(context.ISI['units'] / context.dt)
    trace_len = int((context.sim_duration['clustered'] - context.equilibrate) / context.dt) + baseline_len
    for syn_group in syn_indexes:
        if syn_group not in traces:
            traces[syn_group] = {}
        for num_syns in syn_indexes[syn_group]:
            if num_syns not in traces[syn_group]:
                traces[syn_group][num_syns] = {}
            for count, syn_id in enumerate(syn_indexes[syn_group][num_syns]):
                start = baseline_len + int(count * context.ISI['clustered'] / context.dt)
                end = start + unitary_len
                for rec_name, this_trace in \
                        current_features['unitary_EPSP_traces'][syn_group][syn_condition][syn_id].iteritems():
                    if rec_name not in traces[syn_group][num_syns]:
                        traces[syn_group][num_syns][rec_name] = np.zeros(trace_len)
                    traces[syn_group][num_syns][rec_name][start:end] += this_trace[baseline_len:]
    return traces


def offset_vm(description, vm_target=None):
    """

    :param description: str
    :param vm_target: float
    """
    if vm_target is None:
        vm_target = context.v_init
    step_stim_index = context.sim.get_stim_index('step')
    offset_stim_index = context.sim.get_stim_index('offset')
    context.sim.modify_stim(step_stim_index, amp=0.)
    node = context.rec_nodes[description]
    loc = context.rec_locs[description]
    rec_dict = context.sim.get_rec(description)
    context.sim.modify_stim(offset_stim_index, node=node, loc=loc, amp=0.)
    rec = rec_dict['vec']
    offset = True

    equilibrate = context.equilibrate
    dt = context.dt
    duration = context.duration

    context.sim.tstop = equilibrate
    t = np.arange(0., equilibrate, dt)
    context.sim.modify_stim(offset_stim_index, amp=context.i_holding[description])
    context.sim.run(vm_target)
    vm = np.interp(t, context.sim.tvec, rec)
    v_rest = np.mean(vm[int((equilibrate - 3.)/dt):int((equilibrate - 1.)/dt)])
    initial_v_rest = v_rest
    if v_rest < vm_target - 0.5:
        context.i_holding[description] += 0.01
        while offset:
            if context.sim.verbose:
                print 'increasing i_holding to %.3f (%s)' % (context.i_holding[description], description)
            context.sim.modify_stim(offset_stim_index, amp=context.i_holding[description])
            context.sim.run(vm_target)
            vm = np.interp(t, context.sim.tvec, rec)
            v_rest = np.mean(vm[int((equilibrate - 3.)/dt):int((equilibrate - 1.)/dt)])
            if v_rest < vm_target - 0.5:
                context.i_holding[description] += 0.01
            else:
                offset = False
    elif v_rest > vm_target + 0.5:
        context.i_holding[description] -= 0.01
        while offset:
            if context.sim.verbose:
                print 'decreasing i_holding to %.3f (%s)' % (context.i_holding[description], description)
            context.sim.modify_stim(offset_stim_index, amp=context.i_holding[description])
            context.sim.run(vm_target)
            vm = np.interp(t, context.sim.tvec, rec)
            v_rest = np.mean(vm[int((equilibrate - 3.)/dt):int((equilibrate - 1.)/dt)])
            if v_rest > vm_target + 0.5:
                context.i_holding[description] -= 0.01
            else:
                offset = False
    context.sim.tstop = duration
    return v_rest


def get_spike_shape(vm, spike_times):
    """

    :param vm: array
    :param spike_times: array
    :return: tuple of float: (v_peak, th_v, ADP, AHP)
    """
    equilibrate = context.equilibrate
    dt = context.dt
    th_dvdt = context.th_dvdt

    start = int((equilibrate+1.)/dt)
    vm = vm[start:]
    dvdt = np.gradient(vm, dt)
    th_x = np.where(dvdt > th_dvdt)[0]
    if th_x.any():
        th_x = th_x[0] - int(1.6/dt)
    else:
        th_x = np.where(vm > -30.)[0][0] - int(2./dt)
    th_v = vm[th_x]
    v_before = np.mean(vm[th_x-int(0.1/dt):th_x])
    v_peak = np.max(vm[th_x:th_x+int(5./dt)])
    x_peak = np.where(vm[th_x:th_x+int(5./dt)] == v_peak)[0][0]
    if len(spike_times) > 1:
        end = max(th_x + x_peak + int(2./dt), int((spike_times[1] - 4.) / dt) - start)
    else:
        end = len(vm)
    v_AHP = np.min(vm[th_x+x_peak:end])
    x_AHP = np.where(vm[th_x+x_peak:end] == v_AHP)[0][0]
    AHP = v_before - v_AHP
    # if spike waveform includes an ADP before an AHP, return the value of the ADP in order to increase error function
    ADP = 0.
    rising_x = np.where(dvdt[th_x+x_peak+1:th_x+x_peak+x_AHP-1] > 0.)[0]
    if rising_x.any():
        v_ADP = np.max(vm[th_x+x_peak+1+rising_x[0]:th_x+x_peak+x_AHP])
        pre_ADP = np.mean(vm[th_x+x_peak+1+rising_x[0] - int(0.1/dt):th_x+x_peak+1+rising_x[0]])
        ADP += v_ADP - pre_ADP
    falling_x = np.where(dvdt[th_x + x_peak + x_AHP + 1:end] < 0.)[0]
    if falling_x.any():
        v_ADP = np.max(vm[th_x + x_peak + x_AHP + 1: th_x + x_peak + x_AHP + 1 + falling_x[0]])
        ADP += v_ADP - v_AHP
    return v_peak, th_v, ADP, AHP


def update_AMPA_NMDA(x, local_context=None):
    """

    :param x: array. Default: value=1.711e-03, slope=6.652e-05, tau=7.908e+01
    :param local_context: :class:'Context'
    """
    if local_context is None:
        local_context = context
    cell = local_context.cell
    param_indexes = local_context.param_indexes
    cell.modify_mech_param('apical', 'excitatory synapse', 'gmax',
                           value=x[param_indexes['AMPA.g0']], origin='soma', syn_type='AMPA_KIN',
                           slope=x[param_indexes['AMPA.slope']], tau=x[param_indexes['AMPA.tau']])
    cell.modify_mech_param('apical', 'excitatory synapse', 'gmax', origin='parent', syn_type='AMPA_KIN',
                           custom={'method': 'custom_inherit_by_branch_order', 'branch_order':
                               x[param_indexes['AMPA.gmax bo']]}, replace=False)
    cell.modify_mech_param('apical', 'excitatory synapse', 'gmax', value=x[param_indexes['NMDA.gmax']],
                           syn_type=local_context.NMDA_type)
    cell.modify_mech_param('apical', 'excitatory synapse', 'gamma', value=x[param_indexes['NMDA.gamma']],
                           syn_type=local_context.NMDA_type)
    cell.modify_mech_param('apical', 'excitatory synapse', 'Kd', value=x[param_indexes['NMDA.Kd']],
                           syn_type=local_context.NMDA_type)
    cell.modify_mech_param('apical', 'excitatory synapse', 'kin_scale', value=x[param_indexes['NMDA.kin_scale']],
                           syn_type=local_context.NMDA_type)
    cell.init_synaptic_mechanisms()
    slope = (x[param_indexes['dend.gkabar']] - x[param_indexes['soma.gkabar']]) / 300.
    for sec_type in ['apical']:
        cell.modify_mech_param(sec_type, 'nas', 'gbar', x[param_indexes['dend.gbar_nas']])
        cell.modify_mech_param(sec_type, 'nas', 'gbar', origin='parent', slope=x[param_indexes['dend.gbar_nas slope']],
                               min=x[param_indexes['dend.gbar_nas min']],
                               custom={'method': 'custom_gradient_by_branch_order', 'branch_order':
                                   x[param_indexes['dend.gbar_nas bo']]}, replace=False)
        cell.modify_mech_param(sec_type, 'nas', 'gbar', origin='parent',
                               slope=x[param_indexes['dend.gbar_nas slope']], min=x[param_indexes['dend.gbar_nas min']],
                               custom={'method': 'custom_gradient_by_terminal'}, replace=False)
        cell.modify_mech_param(sec_type, 'kap', 'gkabar', origin='soma', min_loc=75., value=0.)
        cell.modify_mech_param(sec_type, 'kap', 'gkabar', origin='soma', max_loc=75., slope=slope, replace=False)
        cell.modify_mech_param(sec_type, 'kad', 'gkabar', origin='soma', max_loc=75., value=0.)
        cell.modify_mech_param(sec_type, 'kad', 'gkabar', origin='soma', min_loc=75., max_loc=300., slope=slope,
                               value=(x[param_indexes['soma.gkabar']] + slope * 75.), replace=False)
        cell.modify_mech_param(sec_type, 'kad', 'gkabar', origin='soma', min_loc=300.,
                               value=(x[param_indexes['soma.gkabar']] + slope * 300.), replace=False)


def export_sim_results():
    """
    Export the most recent time and recorded waveforms from the QuickSim object.
    """
    with h5py.File(context.temp_output_path, 'a') as f:
        context.sim.export_to_file(f)


def plot_GC_synaptic_integration_features(processed_export_file_path=None):
    """

    :param processed_export_file_path: str
    """
    if processed_export_file_path is None:
        processed_export_file_path = context.processed_export_file_path
    from matplotlib import cm

    with h5py.File(processed_export_file_path, 'r') as f:
        description = 'compound_EPSP_features'
        clustered_branch_names = [key for key in f[description]['data']]
        syn_conditions = f[description]['data'][clustered_branch_names[0]].keys()
        rec_names = [key for key in
                     f[description]['data'][clustered_branch_names[0]][syn_conditions[0]]['traces'].itervalues().next()]
        colors = list(cm.Paired(np.linspace(0, 1, len(syn_conditions))))
        for branch in clustered_branch_names:
            for rec in rec_names:
                fig, axes = plt.subplots(2, 2, sharey=True)
                fig.suptitle('Branch: %s, Rec: %s' % (branch, rec))
                ordered_syn_conditions = ['expected', 'control'] + [syn_condition for syn_condition in syn_conditions if
                                                                    syn_condition not in ['expected', 'control']]
                for i, syn_condition in enumerate(ordered_syn_conditions):
                    col = i / 2
                    row = i % 2
                    for num_syns in f[description]['data'][branch][syn_condition]['traces']:
                        axes[col][row].plot(f[description]['time'],
                                            f[description]['data'][branch][syn_condition]['traces'][num_syns][rec],
                                            c='k')
                    axes[col][row].set_xlabel('Time (ms)')
                    axes[col][row].set_title(syn_condition)
                axes[0][0].set_ylabel('Compound EPSP amplitude (mV)')
                clean_axes(axes)
                fig.tight_layout()
                fig.subplots_adjust(top=0.85)

        fig, axes = plt.subplots(1, len(clustered_branch_names), sharey=True)
        if len(clustered_branch_names) == 1:
            axes = [axes]
        fig.suptitle('Rec: soma')
        diagonal = np.linspace(0., np.max(f[description]['data'][branch]['expected']['EPSP_amp']), 10)
        for i, branch in enumerate(clustered_branch_names):
            for j, syn_condition in enumerate((syn_condition for syn_condition in syn_conditions if
                                               syn_condition != 'expected')):
                axes[i].plot(f[description]['data'][branch]['expected']['EPSP_amp'],
                             f[description]['data'][branch][syn_condition]['EPSP_amp'],
                             c=colors[j], label=syn_condition)
                axes[i].set_title('Branch: %s' % branch)
                axes[i].set_xlabel('Expected EPSP amp (mV)')
            axes[i].plot(diagonal, diagonal, c='lightgrey', linestyle='--')
        axes[0].set_ylabel('Actual EPSP amp (mV)')
        axes[0].legend(loc='best', frameon=False, framealpha=0.5)
        clean_axes(axes)
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)

        description = 'unitary_EPSP_features'
        fig, axes = plt.subplots(1, len(rec_names))
        if len(rec_names) == 1:
            axes = [axes]
        for i, rec in enumerate(rec_names):
            for j, syn_condition in enumerate(f[description]['data']):
                axes[i].plot(f[description]['time'], f[description]['data'][syn_condition]['mean_traces'][rec],
                             c=colors[j], label=syn_condition)
                axes[i].set_xlabel('Time (ms)')
                axes[i].set_title(rec)
        axes[0].set_ylabel('Unitary EPSP amplitude (mV)')
        axes[0].legend(loc='best', frameon=False, framealpha=0.5)
        clean_axes(axes)
        fig.tight_layout()
    plt.show()
    plt.close()
