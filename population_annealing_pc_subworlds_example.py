from nested.parallel import *
from nested.optimize_utils import *


context = Context()


def complex_problem(parameters, model_id=None, export=False):
    """

    :param parameters: array
    :param model_id: int or str
    :param export: bool
    :return: dict
    """
    print('Process: %i; model_id: %s; evaluating parameters: %s' %
          (os.getpid(), str(model_id), ', '.join('%.3f' % x for x in parameters)))
    sys.stdout.flush()

    # Test handling of failure to compute required feature
    if parameters[0] > 1.:
        return dict()

    features = {}
    num_params = len(parameters)
    f1 = parameters[0]
    features['f1'] = f1
    g = 1. + 9. / (num_params - 1.) * np.sum(parameters[1:])
    features['g'] = g
    h = 1. - np.sqrt(f1 / g)
    features['h'] = h

    return features


def get_objectives(features, model_id=None, export=False):
    """

    :param features: dict
    :param model_id: int or str
    :param export: bool
    :return: tuple of dict
    """
    objectives = {}
    for feature_name in ['f1', 'g', 'h']:
        if feature_name not in features:
            return dict(), dict()
        objective_name = feature_name
        objectives[objective_name] = features[feature_name]
    f2 = features['g'] * features['h']
    objectives['f2'] = f2
    return features, objectives


def test_shared_features(parameters, model_id=None, export=False):
    """

    :param parameters: array
    :param model_id: int or str
    :param export: bool
    :return: dict
    """
    return {'shared_features': 1.}


def config_synchronize():
    pid = os.getpid()
    interface = pc_find_interface()
    global_rank = interface.global_comm.rank
    global_size = interface.global_comm.size
    local_rank = interface.comm.rank
    local_size = interface.comm.size
    interface.global_comm.barrier()
    if context.disp:
        print('pid: %i with global MPI rank %i / %i and local MPI rank %i / %i completed config_synchronize at %s' %
              (pid, global_rank, global_size, local_rank, local_size, datetime.datetime.now()))
        sys.stdout.flush()
