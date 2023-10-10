import os
import sys
import numpy as np
from nested.utils import Context, param_array_to_dict
from nested.optimize_utils import update_source_contexts


context = Context()


def config_controller():
    if 'controller_comm' in context() and context.controller_comm is not None and context.disp:
        print('context.controller_comm is defined on controller with size: %i' % context.controller_comm.size)
        sys.stdout.flush()


def config_worker():
    context.updated = False
    if 'comm' in context() and context.comm is not None and context.disp:
        print('pid: %i; context.comm is defined on worker with size: %i' % (os.getpid(), context.comm.size))
        sys.stdout.flush()


def test_update_context(x, context):
    context.update(param_array_to_dict(x, context.param_names))
    if context.disp:
        print('pid: %i; context updated' % os.getpid())
        sys.stdout.flush()


def collective():
    test = None
    if context.global_comm.rank == 0:
        test = 0
    test = context.global_comm.bcast(test, root=0)
    results = context.global_comm.gather(test, root=0)
    if context.global_comm.rank == 0:
        print('pid: %i; results gathered from collective operation: %s' % (os.getpid(), results))
        sys.stdout.flush()


def complex_problem(parameters, model_id=None, export=False, plot=False):
    """
    Multi-objective optimization benchmark problem from:
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    Zitzler–Deb–Thiele's function N. 1
    :param parameters: array
    :param model_id: int or str
    :param export: bool
    :return: dict
    """
    print('Process: %i; model_id: %s; evaluating parameters: %s' %
          (os.getpid(), str(model_id), ', '.join('%.3f' % x for x in parameters)))
    sys.stdout.flush()

    update_source_contexts(parameters, context)

    # Test handling of failure to compute required feature
    if context.p0 > 1.:
        return dict()

    features = {}
    num_params = len(parameters)
    f1 = context.p0
    features['f1'] = f1
    g = 1. + 9. / (num_params - 1.) * np.sum(parameters[1:])
    features['g'] = g
    h = 1. - np.sqrt(f1 / g)
    features['h'] = h

    return features


def get_objectives(features, model_id=None, export=False, plot=False):
    """

    :param features: dict
    :param model_id: int or str
    :param export: bool
    :return: tuple of dict
    """
    objectives = {}
    for feature_name in features:
        objectives[feature_name] = features[feature_name]
    f2 = features['g'] * features['h']
    objectives['f2'] = f2
    return features, objectives


def test_shared_features(parameters, model_id=None, export=False, plot=False):
    """

    :param parameters: array
    :param model_id: int or str
    :param export: bool
    :return: dict
    """
    if 'shared_feature_kwarg' in context():
        val = context.shared_feature_kwarg
    else:
        val = 1
    return {'shared_features': val}
