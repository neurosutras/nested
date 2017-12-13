from mpi4py import MPI
from neuron import h
from ring_test_voltage import *
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moopgen import *


context = Context()


def config_engine(comm, subworld_size, target_val, target_range, **kwargs):
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
    context.update(locals())
    set_constants()
    pc = h.ParallelContext()
    pc.subworlds(subworld_size)
    context.pc = pc
    setup_network(**kwargs)
    print 'setup network on MPI rank %d' %context.comm.rank
    #context.pc.barrier()
    context.pc.runworker()


def report_pc_id():
    return {'MPI rank': context.comm.rank, 'pc.id_world': context.pc.id_world(), 'pc.id': context.pc.id()}

def set_constants():
    """

    """
    ncell = 2
    delay = 1
    tstop = 100
    context.update(locals())

def setup_network(verbose=False, cvode=False, daspk=False, **kwargs):
    """

    :param verbose: bool
    :param cvode: bool
    :param daspk: bool
    """
    context.ring = Ring(context.ncell, context.delay, context.pc)
    # context.pc.set_maxstep(10)


def get_EPSP_features(indivs):
    print 'get_feature rank %i active' %context.pc.id_world()
    features = []
    for indiv in indivs:
        context.pc.submit(calc_EPSP, indiv)
    while context.pc.working():
        features.append(context.pc.pyret())
    return features


def get_objectives(features):
    objectives = {}
    for i, feature in enumerate(features):
        if feature is None:
            objectives[i] = None
        else:
            objectives[i] = {'EPSP': feature['EPSP'] - context.target_val['EPSP']}
    return features, objectives


def calc_EPSP(indiv):
    weight = indiv['x']
    context.ring.update_syn_weight(weight)
    results = runring(context.ring)
    max_ind = np.argmax(np.array(results['rec'][1]))
    """
    if context.comm.rank == 0:
        print results
    """
    processed_result = {'EPSP': results['rec'][1][max_ind], 'peak_t': results['t'][1][max_ind]}
    return {'pop_id': indiv['pop_id'], 'result_list': [{'id': context.pc.id_world()}, processed_result]}

"""
def calc_spike_count(indiv, i):
    """"""
    x = indiv['x']
    results = runring(context.ring)
    return {'pop_id': int(i), 'result_list': [{'id': context.pc.id_world()}, results]}
"""

def end_optimization():
    context.pc.done()