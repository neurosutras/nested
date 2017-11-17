__author__ = 'Aaron D. Milstein'
from specify_cells4 import *
from plot_results import *
from moopgen import *

"""
Submodule used by parallel_optimize to tune integrative properties of dentate granule cells to match target features of 
somatodendritic filtering of EPSPs (including TTX sensitivity).
Requires a YAML file to specify required configuration parameters. 
Requires use of an ipyparallel client.
"""

context = Context()


def config_interactive(config_file_path='data/parallel_optimize_GC_EPSC_attenuation_config.yaml', output_dir='data',
                       temp_output_path=None, export_file_path=None, verbose=True, disp=True):
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

    """
    seed_offset = 8. * 2e6
    num_branches = 2
    branch_names = ['branch%i' % i for i in xrange(num_branches)]
    syn_conditions = ['control', 'TTX']
    ISI = {'long': 100., 'short': 10.}  # inter-stimulus interval for synaptic stim (ms)
    units_per_sim = {'long': 1, 'short': 5}
    equilibrate = 250.  # time to steady-state
    stim_dur = 150.
    sim_duration = {'long': equilibrate + units_per_sim['long'] * ISI['long'] + 50,
                    'short': equilibrate + units_per_sim['short'] * ISI['short'] + 150.,
                    'default': equilibrate + stim_dur}
    trace_baseline = 10.
    duration = max(sim_duration.values())
    th_dvdt = 10.
    dt = 0.02
    v_init = -77.
    v_active = -77.
    syn_types = ['EPSC']
    local_random = random.Random()
    i_holding = {'soma': 0.}
    i_th = {'soma': 0.1}

    target_iEPSP_amp = 3.
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

    # Choose apical branches to measure attenuation of EPSPs generated by dendritic injection of EPSC-shaped currents.
    # Each branch must be > 100 um from the soma.
    candidate_branches = [apical for apical in cell.apical if
                          (75. < cell.get_distance_to_node(cell.tree.root, apical) < 150.) and apical.sec.L > 50.]
    context.local_random.shuffle(candidate_branches)

    syn_list = []
    i = 0
    # Synapses must be 25 um from the branch point.
    while len(syn_list) < context.num_branches and i < len(candidate_branches):
        branch = candidate_branches[i]
        parents = [syn.branch.parent for syn in syn_list]
        if branch.parent not in parents:
            syn = Synapse(context.cell, branch, loc=25./branch.sec.L, syn_types=context.syn_types, stochastic=False)
            syn_list.append(syn)
        i += 1
    if len(syn_list) < context.num_branches:
        raise Exception('parallel_optimize_GC_EPSC_attenuation: cell with index %i: fewer than target number of '
                        'branches meet criterion.' % context.neuroH5_index)
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
    rec_locs = {'soma': 0., 'dend': dend_loc, 'local_branch': 0.}
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


def get_iEPSP_features_long_ISI(indiv, c=None, client_range=None, export=False):
    """
    Distribute simulations across available engines for measuring iEPSP amplitude.
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
    ISI_key = 'long'
    syn_condition = 'control'
    result = map_func(compute_iEPSP_amp_features, [x] * context.num_branches, range(context.num_branches),
                      [ISI_key] * context.num_branches, [syn_condition] * context.num_branches,
                      [None] * context.num_branches, [export] * context.num_branches)
    return {'pop_id': indiv['pop_id'], 'client_range': client_range, 'async_result': result,
            'filter_features': filter_iEPSP_features}


def compute_iEPSP_amp_features(x, syn_index, ISI_key, syn_condition, imax=None, export=False, plot=False):
    """

    :param x: arr
    :param syn_index: int
    :param ISI_key: str
    :param syn_condition: str
    :param imax: float
    :param export: bool
    :param plot: bool
    :return: dict
    """
    start_time = time.time()
    update_submodule_params(x, context)

    soma_vm = offset_vm('soma', context.v_init)

    if imax is None:
        imax = 0.15
        result = optimize.minimize(iEPSP_amp_error, [imax], args=(syn_index,), method='L-BFGS-B',
                                   bounds=[(imax / 2., imax * 2.)],
                                   options={'ftol': 1e-3, 'disp': True, 'maxfun': 5})
        imax = result.x[0]

    syn = context.syn_list[syn_index]
    syn.source.play(h.Vector([context.equilibrate + i * context.ISI[ISI_key] for
                              i in xrange(context.units_per_sim[ISI_key])]))
    syn.target(context.syn_types[0]).imax = imax
    context.sim.parameters['input_loc'] = syn.branch.type
    context.sim.modify_rec(context.sim.get_rec_index('local_branch'), node=syn.branch, loc=syn.loc)
    description = 'ISI: %s, condition: %s, %s' % (ISI_key, syn_condition, context.branch_names[syn_index])
    duration = context.sim_duration[ISI_key]
    title = 'iEPSP_amp_features'
    context.sim.tstop = duration
    context.sim.parameters['duration'] = duration
    context.sim.parameters['title'] = title
    context.sim.parameters['description'] = description
    if syn_condition == 'TTX':
        context.cell.zero_na()
    context.sim.run(context.v_init)
    dt = context.dt
    equilibrate = context.equilibrate
    interp_t = np.arange(0., duration, dt)
    trace_baseline = context.trace_baseline

    result = {'traces': {ISI_key: {syn_condition: {syn_index: {}}}},
              'imax': {syn_index: imax}}
    start = int(equilibrate / dt)
    end = int(duration / dt)
    trace_start = start - int(trace_baseline / dt)
    baseline_start, baseline_end = int(start - 3. / dt), int(start - 1. / dt)
    for rec in context.sim.rec_list:
        interp_vm = np.interp(interp_t, context.sim.tvec, rec['vec'])
        baseline = np.mean(interp_vm[baseline_start:baseline_end])
        corrected_vm = interp_vm[trace_start:end] - baseline
        result['traces'][ISI_key][syn_condition][syn_index][rec['description']] = np.array(corrected_vm)
    syn.source.play(h.Vector())

    if context.disp:
        print 'Process: %i: %s: %s took %.3f s' % (os.getpid(), title, description, time.time() - start_time)
    if plot:
        context.sim.plot()
    if export:
        export_sim_results()
    return result


def filter_iEPSP_features(computed_result_list, current_features, target_val, target_range, export=False):
    """

    :param computed_result_list: list of dict (each dict contains results from a single simulation)
    :param current_features: dict
    :param target_val: dict of float
    :param target_range: dict of float
    :param export: bool
    :return: dict
    """
    traces = {}
    iEPSP_amp = {}
    imax = {}
    ISI_key = computed_result_list[0]['traces'].iterkeys().next()
    for this_result_dict in computed_result_list:
        for syn_condition in this_result_dict['traces'][ISI_key]:
            if syn_condition not in traces:
                traces[syn_condition] = {}
                iEPSP_amp[syn_condition] = {}
            for syn_index in this_result_dict['traces'][ISI_key][syn_condition]:
                traces[syn_condition][syn_index] = this_result_dict['traces'][ISI_key][syn_condition][syn_index]
                iEPSP_amp[syn_condition][syn_index] = {}
                for rec_name in traces[syn_condition][syn_index]:
                    iEPSP_amp[syn_condition][syn_index][rec_name] = np.max(traces[syn_condition][syn_index][rec_name])
                imax[syn_index] = this_result_dict['imax'][syn_index]
    new_features = {'iEPSP_amp_' + ISI_key: iEPSP_amp,
                    'iEPSP_traces_' + ISI_key: traces,
                    'imax': imax}
    if export:
        description = 'iEPSP_features'
        t = np.arange(-context.trace_baseline, context.sim_duration[ISI_key] - context.equilibrate, context.dt)
        with h5py.File(context.processed_export_file_path, 'a') as f:
            if description not in f:
                f.create_group(description)
            if ISI_key not in f[description]:
                f[description].create_group(ISI_key)
            if 'time' not in f[description][ISI_key]:
                f[description][ISI_key].create_dataset('time', compression='gzip', compression_opts=9, data=t)
            if 'traces' not in f[description][ISI_key]:
                f[description][ISI_key].create_group('traces')
            for syn_condition in traces:
                if syn_condition not in f[description][ISI_key]['traces']:
                    f[description][ISI_key]['traces'].create_group(syn_condition)
                for rec in traces[syn_condition].itervalues().next():
                    this_mean_trace = np.mean([traces[syn_condition][syn_index][rec]
                                               for syn_index in traces[syn_condition]], axis=0)
                    f[description][ISI_key]['traces'][syn_condition].create_dataset(rec, compression='gzip',
                                                                                    compression_opts=9,
                                                                                    data=this_mean_trace)
    return new_features


def get_iEPSP_features_short_ISI(indiv, c=None, client_range=None, export=False):
    """
    Distribute simulations across available engines for measuring iEPSP amplitude.
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
    ISI_key = 'short'
    syn_conditions = ['control'] * context.num_branches + ['TTX'] * context.num_branches
    if 'features' in indiv and 'imax' in indiv['features'] and len(indiv['features']['imax']) > 0:
        imax = [indiv['features']['imax'][syn_index] for syn_index in xrange(len(indiv['features']['imax']))]
    else:
        imax = [None for i in xrange(context.num_branches)]
    result = map_func(compute_iEPSP_amp_features, [x] * 2 * context.num_branches, range(context.num_branches) * 2,
                      [ISI_key] * 2 * context.num_branches, syn_conditions, imax * 2 ,
                      [export] * 2 * context.num_branches)
    return {'pop_id': indiv['pop_id'], 'client_range': client_range, 'async_result': result,
            'filter_features': filter_iEPSP_features}


def get_objectives(features, target_val, target_range):
    """

    :param features: dict
    :param target_val: dict of float
    :param target_range: dict of float
    :return: tuple of dict
    """
    objectives = {}
    objective_names = ['EPSC_attenuation_long_ISI', 'EPSC_attenuation_short_ISI', 'EPSC_amplification_soma',
                       'EPSC_amplification_dend', 'EPSC_attenuation_TTX']
    features['EPSC_attenuation_long_ISI'] = np.mean([features['iEPSP_amp_long']['control'][i]['soma'] /
                                                     features['iEPSP_amp_long']['control'][i]['local_branch'] for
                                                     i in xrange(context.num_branches)])
    features['EPSC_attenuation_short_ISI'] = np.mean([features['iEPSP_amp_short']['control'][i]['soma'] /
                                                      features['iEPSP_amp_short']['control'][i]['local_branch'] for
                                                      i in xrange(context.num_branches)])
    features['EPSC_amplification_soma'] = np.mean([features['iEPSP_amp_short']['control'][i]['soma'] /
                                                   features['iEPSP_amp_long']['control'][i]['soma'] for
                                                   i in xrange(context.num_branches)])
    features['EPSC_amplification_dend'] = np.mean([features['iEPSP_amp_short']['control'][i]['local_branch'] /
                                                   features['iEPSP_amp_long']['control'][i]['local_branch'] for
                                                   i in xrange(context.num_branches)])
    features['EPSC_attenuation_TTX'] = np.mean([features['iEPSP_amp_short']['TTX'][i]['soma'] /
                                                features['iEPSP_amp_short']['control'][i]['soma'] for
                                                i in xrange(context.num_branches)])
    for objective_name in objective_names:
        objectives[objective_name] = ((target_val[objective_name] - features[objective_name]) /
                                      target_range[objective_name]) ** 2.
    return features, objectives


def iEPSP_amp_error(x, syn_index):
    """

    :param x: array
    :param syn_index: int
    :return: float
    """
    start_time = time.time()
    syn = context.syn_list[syn_index]
    syn.target(context.syn_types[0]).imax = x[0]
    syn.source.play(h.Vector([context.equilibrate]))
    context.sim.tstop = context.equilibrate + context.ISI['long']
    context.sim.run(context.v_init)
    syn.source.play(h.Vector())
    rec = context.sim.get_rec('soma')['vec']
    t = np.arange(0., context.duration, context.dt)
    vm = np.interp(t, context.sim.tvec, rec)
    baseline = np.mean(vm[int((context.equilibrate - 3.) / context.dt):int((context.equilibrate - 1.) / context.dt)])
    vm -= baseline
    iEPSP_amp = np.max(vm[int(context.equilibrate / context.dt):])
    Err = ((iEPSP_amp - context.target_iEPSP_amp) / 0.01) ** 2.
    if context.disp:
        print '%s.imax: %.4f, soma iEPSP amp: %.3f, simulation took %.1f s' % (context.syn_types[0], x[0], iEPSP_amp,
                                                                               time.time() - start_time)
    return Err


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


def update_nap_params(x, local_context=None):
    """
    :param x: array ['soma.gbar_nas', 'dend.gbar_nas', 'dend.gbar_nas slope', 'dend.gbar_nas min', 'dend.gbar_nas bo',
                    'axon.gbar_nax', 'ais.gbar_nax', 'soma.gkabar', 'dend.gkabar', 'soma.gkdrbar', 'axon.gkabar',
                    'soma.sh_nas/x', 'soma.sha_nas/x', 'ais.sha_nax', 'soma.gCa factor', 'soma.gCadepK factor',
                    'soma.gkmbar', 'ais.gkmbar']
    """
    if local_context is None:
        local_context = context
    cell = local_context.cell
    param_indexes = local_context.param_indexes
    cell.modify_mech_param('soma', 'nas', 'sha', x[param_indexes['soma.sha_nas/x']])
    for sec_type in ['apical']:
        cell.modify_mech_param(sec_type, 'nas', 'sha', origin='soma')
    cell.modify_mech_param('axon_hill', 'nax', 'sha', x[param_indexes['soma.sha_nas/x']])
    cell.modify_mech_param('axon', 'nax', 'sha', origin='axon_hill')
    cell.modify_mech_param('ais', 'nax', 'sha', x[param_indexes['ais.sha_nax']] + x[param_indexes['soma.sha_nas/x']])


def update_spike_shape_params(x, local_context=None):
    """
    :param x: array ['soma.gbar_nas', 'dend.gbar_nas', 'dend.gbar_nas slope', 'dend.gbar_nas min', 'dend.gbar_nas bo',
                    'axon.gbar_nax', 'ais.gbar_nax', 'soma.gkabar', 'dend.gkabar', 'soma.gkdrbar', 'axon.gkabar',
                    'soma.sh_nas/x', 'ais.sha_nax', 'soma.gCa factor', 'soma.gCadepK factor', 'soma.gkmbar',
                    'ais.gkmbar']
    """
    if local_context is None:
        local_context = context
    cell = local_context.cell
    param_indexes = local_context.param_indexes
    cell.modify_mech_param('soma', 'nas', 'gbar', x[param_indexes['soma.gbar_nas']])
    cell.modify_mech_param('soma', 'kdr', 'gkdrbar', x[param_indexes['soma.gkdrbar']])
    cell.modify_mech_param('soma', 'kap', 'gkabar', x[param_indexes['soma.gkabar']])
    slope = (x[param_indexes['dend.gkabar']] - x[param_indexes['soma.gkabar']]) / 300.
    cell.modify_mech_param('soma', 'nas', 'sh', x[param_indexes['soma.sh_nas/x']])
    for sec_type in ['apical']:
        cell.reinitialize_subset_mechanisms(sec_type, 'nas')
        cell.modify_mech_param(sec_type, 'kap', 'gkabar', origin='soma', min_loc=75., value=0.)
        cell.modify_mech_param(sec_type, 'kap', 'gkabar', origin='soma', max_loc=75., slope=slope, replace=False)
        cell.modify_mech_param(sec_type, 'kad', 'gkabar', origin='soma', max_loc=75., value=0.)
        cell.modify_mech_param(sec_type, 'kad', 'gkabar', origin='soma', min_loc=75., max_loc=300., slope=slope,
                               value=(x[param_indexes['soma.gkabar']] + slope * 75.), replace=False)
        cell.modify_mech_param(sec_type, 'kad', 'gkabar', origin='soma', min_loc=300.,
                               value=(x[param_indexes['soma.gkabar']] + slope * 300.), replace=False)
        cell.modify_mech_param(sec_type, 'kdr', 'gkdrbar', origin='soma')
        cell.modify_mech_param(sec_type, 'nas', 'sha', 0.)  # 5.)
        cell.modify_mech_param(sec_type, 'nas', 'gbar',
                               x[param_indexes['dend.gbar_nas']])
        cell.modify_mech_param(sec_type, 'nas', 'gbar', origin='parent', slope=x[param_indexes['dend.gbar_nas slope']],
                               min=x[param_indexes['dend.gbar_nas min']],
                               custom={'method': 'custom_gradient_by_branch_order',
                                       'branch_order': x[param_indexes['dend.gbar_nas bo']]}, replace=False)
        cell.modify_mech_param(sec_type, 'nas', 'gbar', origin='parent',
                               slope=x[param_indexes['dend.gbar_nas slope']], min=x[param_indexes['dend.gbar_nas min']],
                               custom={'method': 'custom_gradient_by_terminal'}, replace=False)
    cell.reinitialize_subset_mechanisms('axon_hill', 'kap')
    cell.reinitialize_subset_mechanisms('axon_hill', 'kdr')
    cell.modify_mech_param('ais', 'kdr', 'gkdrbar', origin='soma')
    cell.modify_mech_param('ais', 'kap', 'gkabar', x[param_indexes['axon.gkabar']])
    cell.modify_mech_param('axon', 'kdr', 'gkdrbar', origin='ais')
    cell.modify_mech_param('axon', 'kap', 'gkabar', origin='ais')
    cell.modify_mech_param('axon_hill', 'nax', 'sh', x[param_indexes['soma.sh_nas/x']])
    cell.modify_mech_param('axon_hill', 'nax', 'gbar', x[param_indexes['soma.gbar_nas']])
    cell.modify_mech_param('axon', 'nax', 'gbar', x[param_indexes['axon.gbar_nax']])
    for sec_type in ['ais', 'axon']:
        cell.modify_mech_param(sec_type, 'nax', 'sh', origin='axon_hill')
    cell.modify_mech_param('soma', 'Ca', 'gcamult', x[param_indexes['soma.gCa factor']])
    cell.modify_mech_param('soma', 'CadepK', 'gcakmult', x[param_indexes['soma.gCadepK factor']])
    cell.modify_mech_param('soma', 'km3', 'gkmbar', x[param_indexes['soma.gkmbar']])
    cell.modify_mech_param('ais', 'km3', 'gkmbar', x[param_indexes['ais.gkmbar']])
    cell.modify_mech_param('axon_hill', 'km3', 'gkmbar', origin='soma')
    cell.modify_mech_param('axon', 'km3', 'gkmbar', origin='ais')
    cell.modify_mech_param('ais', 'nax', 'sha', x[param_indexes['ais.sha_nax']])
    cell.modify_mech_param('ais', 'nax', 'gbar', x[param_indexes['ais.gbar_nax']])


def export_sim_results():
    """
    Export the most recent time and recorded waveforms from the QuickSim object.
    """
    with h5py.File(context.temp_output_path, 'a') as f:
        context.sim.export_to_file(f)


def plot_GC_EPSC_attenuation_features(processed_export_file_path=None):
    """

    :param processed_export_file_path: str
    """
    if processed_export_file_path is None:
        processed_export_file_path = context.processed_export_file_path
    from matplotlib import cm

    with h5py.File(processed_export_file_path, 'r') as f:
        fig, axes = plt.subplots(1, 3, sharey=True)
        description = 'iEPSP_features'
        rec_names = ['soma', 'local_branch']
        colors = list(cm.Paired(np.linspace(0, 1, len(rec_names))))
        axes[0].set_ylabel('iEPSP amplitude (mV)')
        for j in xrange(len(axes)):
            axes[j].set_xlabel('Time (ms)')
        axes[0].set_title('Long ISI - Control')
        axes[1].set_title('Short ISI - Control')
        axes[2].set_title('Short ISI - TTX')
        for i, rec in enumerate(rec_names):
            axes[0].plot(f[description]['long']['time'], f[description]['long']['traces']['control'][rec], label=rec,
                         c=colors[i])
            axes[1].plot(f[description]['short']['time'], f[description]['short']['traces']['control'][rec], label=rec,
                         c=colors[i])
            axes[2].plot(f[description]['short']['time'], f[description]['short']['traces']['TTX'][rec], label=rec,
                         c=colors[i])
    clean_axes(axes)
    fig.tight_layout()
    plt.show()
    plt.close()
