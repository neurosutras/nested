from mpi4py import MPI
from neuron import h
from ring import *
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moopgen import *


context = Context()

def config_controller(**kwargs):
    """

    :param export_file_path: str (path)
    """
    context.update(kwargs)
    set_constants()

def controller_details():
    print context()

def config_engine(comm, **kwargs):
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
    pc.subworlds(2)
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


def get_feature(indivs):
    print 'get_feature rank %i active' %context.pc.id_world()
    features = []
    for i, indiv in enumerate(indivs):
        context.pc.submit(calc_spike_count, indiv, i)
    while context.pc.working():
        features.append(context.pc.pyret())

    return features


def calc_spike_count(indiv, i):
    """"""
    x = indiv['x']
    results = runring(context.ring)
    return {'pop_id': int(i), 'result_list': [{'id': context.pc.id_world()}, results]}


def end_optimization():
    context.pc.done()