from mpi4py import MPI
from neuron import h
from optimize_utils import *
from ring import *

context = Context()

def config_controller(**kwargs):
    """

    :param export_file_path: str (path)
    """
    context.update(kwargs)
    init_context()

def config_engine(**kwargs):
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
    context.update(kwargs)
    init_context()
    pc = h.ParallelContext()
    pc.subworlds(1)
    context.pc = pc
    comm = MPI.COMM_WORLD
    rank = comm.rank
    context.update(locals())
    setup_network(context.pc, **kwargs)
    print 'setup network'
    # pc.runworker()

def report_pc_id(client_id):
    return {'client_id': client_id, 'MPI rank': context.rank, 'pc.id_world': context.pc.id_world(), 'pc.id': context.pc.id()}

def init_context():
    """

    """
    ncell = 5
    delay = 1
    tstop = 100
    context.update(locals())

def setup_network(pc, verbose=False, cvode=False, daspk=False, **kwargs):
    """

    :param verbose: bool
    :param cvode: bool
    :param daspk: bool
    """
    # from ring import Ring
    context.ring = Ring(context.ncell, context.delay, pc)
    # context.pc.set_maxstep(10)

#def get_feature(indiv, c, client_range, export=False):
def get_feature(indivs, c):
    result = c[:].map_async(compute_feature, indivs)
    while not result.ready():
        for stdout in result.stdout:
            if stdout:
                for line in stdout.splitlines():
                    print line
        sys.stdout.flush()
        time.sleep(1.)
    print result.get()

def compute_feature(indiv):
    x = indiv['x']
    context.pc.runworker()
    context.pc.submit(calc_spike_count, x)
    features = []
    while (context.pc.working()):
        result = context.pc.pyret
        features.append(result)
    context.pc.done()
    return features


def calc_spike_count(x):
    """"""
    context.ring.runring()
    """
    #update_submodule_params(x, context)
    h.stdinit()
    context.pc.psolve(context.tstop)
    spkcnt = context.pc.allreduce(len(context.ring.tvec), 1)
    tmax = context.pc.allreduce(context.ring.tvec.x[-1], 2)
    tt = h.Vector()
    idv = h.Vector()
    context.pc.allgather(context.ring.tvec.x[-1], tt)
    context.pc.allgather(context.ring.idvec.x[-1], idv)
    idmax = int(idv.x[int(tt.max_ind())])
    return (int(spkcnt), tmax, idmax, (context.ncell, context.delay, context.tstop, (context.pc.id_world(),
                                                                                     context.pc.nhost_world())))
    """