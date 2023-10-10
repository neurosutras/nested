import os
import sys
import numpy as np
from nested.utils import Context


context = Context()


def config_controller():
    if 'controller_comm' in context():
        if context.disp:
            print('context.controller_comm is defined on controller with size: %i' % context.controller_comm.size)
            sys.stdout.flush()
    else:
        raise RuntimeError('config_controller: context.controller_comm is not defined')


def config_worker():
    if 'comm' in context():
        if context.disp:
            print('context.comm is defined on worker rank: %i with size: %i' % (context.comm.rank, context.comm.size))
            sys.stdout.flush()
    else:
        raise RuntimeError('config_worker: context.comm is not defined on a worker')


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
    if context.disp:
        print('Process: %i; model_id: %s; evaluating parameters: %s' %
              (os.getpid(), str(model_id), ', '.join('%.3f' % x for x in parameters)))
        sys.stdout.flush()

    features = {}
    num_params = len(parameters)
    f1 = parameters[0]
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
