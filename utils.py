__author__ = 'Aaron D. Milstein and Grace Ng'
import h5py
import math
import pickle
import datetime
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.mlab as mm
import scipy.optimize as optimize
import scipy.signal as signal
import random
import pprint
import sys
import os
import gc


data_dir = 'data/'


def write_to_pkl(file_path, data):
    """
    Export a python object to .pkl
    :param file_path: str
    :param data: picklable object
    """
    output = open(file_path, 'wb')
    pickle.dump(data, output, 2)
    output.close()


def read_from_pkl(file_path):
    """
    Import a python object from .pkl
    :param file_path: str
    :return: unpickled object
    """
    if os.path.isfile(file_path):
        pkl_file = open(file_path, 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()
        return data
    else:
        raise Exception('File: {} does not exist.'.format(file_path))


def write_to_yaml(file_path, data):
    """
    Export a python dict to .yaml
    :param file_path: str
    :param dict: dict
    """
    import yaml
    with open(file_path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def read_from_yaml(file_path):
    """
    Import a python dict from .yaml
    :param file_path: str (should end in '.yaml')
    :return:
    """
    import yaml
    if os.path.isfile(file_path):
        with open(file_path, 'r') as stream:
            data = yaml.load(stream)
        return data
    else:
        raise Exception('File: {} does not exist.'.format(file_path))


def merge_hdf5_files(file_path_list, new_file_path=None, verbose=True):
    """
    Combines the contents of multiple .hdf5 files.
    :param file_path_list: list of str (paths)
    :param new_file_path: str (path)
    :return str (path)
    """
    if new_file_path is None:
        new_file_path = 'merged_hdf5_'+datetime.datetime.today().strftime('%m%d%Y%H%M')+'_'+os.getpid()
    new_f = h5py.File(new_file_path, 'w')
    iter = 0
    for old_file_path in file_path_list:
        old_f = h5py.File(old_file_path, 'r')
        for old_group in old_f.itervalues():
            new_f.copy(old_group, new_f, name=str(iter))
            iter += 1
        old_f.close()
    new_f.close()
    if verbose:
        print 'merge_hdf5_files: exported to file_path: %s' % new_file_path
    return new_file_path


def null_minimizer(fun, x0, *args, **options):
    """
    Rather than allow scipy.optimize.basinhopping to pass each local mimimum to a gradient descent algorithm for
    polishing, this method catches and passes all local minima so basinhopping can proceed.
    """
    return optimize.OptimizeResult(x=x0, fun=fun(x0, *args), success=True, nfev=1)


def sliding_window(unsorted_x, y=None, bin_size=60., window_size=3, start=-60., end=7560.):
    """
    An ad hoc function used to compute sliding window density and average value in window, if a y array is provided.
    :param unsorted_x: array
    :param y: array
    :return: bin_center, density, rolling_mean: array, array, array
    """
    indexes = range(len(unsorted_x))
    indexes.sort(key=unsorted_x.__getitem__)
    sorted_x = map(unsorted_x.__getitem__, indexes)
    if y is not None:
        sorted_y = map(y.__getitem__, indexes)
    window_dur = bin_size * window_size
    bin_centers = np.arange(start+window_dur/2., end-window_dur/2.+bin_size, bin_size)
    density = np.zeros(len(bin_centers))
    rolling_mean = np.zeros(len(bin_centers))
    x0 = 0
    x1 = 0
    for i, bin in enumerate(bin_centers):
        while sorted_x[x0] < bin - window_dur / 2.:
            x0 += 1
            # x1 += 1
        while sorted_x[x1] < bin + window_dur / 2.:
            x1 += 1
        density[i] = (x1 - x0) / window_dur * 1000.
        if y is not None:
            rolling_mean[i] = np.mean(sorted_y[x0:x1])
    return bin_centers, density, rolling_mean


def clean_axes(axes):
    """
    Remove top and right axes from pyplot axes object.
    :param axes:
    """
    if not type(axes) in [np.ndarray, list]:
        axes = [axes]
    elif type(axes) == np.ndarray:
        axes = axes.flatten()
    for axis in axes:
        axis.tick_params(direction='out')
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.get_xaxis().tick_bottom()
        axis.get_yaxis().tick_left()


def sort_str_list(str_list, seperator='_', end=None):
    """
    Given a list of filenames ending with (separator)int, sort the strings by increasing value of int.
    If there is a suffix at the end of the filename, provide it so it can be ignored.
    :param str_list: list of str
    :param seperator: str
    :param end: str
    :return: list of str
    """
    indexes = range(len(str_list))
    values = []
    for this_str in str_list:
        if end is not None:
            this_str = this_str.split(end)[0]
        this_value = int(this_str.split(seperator)[-1])
        values.append(this_value)
    indexes.sort(key=values.__getitem__)
    sorted_str_list = map(str_list.__getitem__, indexes)
    return sorted_str_list


def list_find(f, items):
    """
    Return index of first instance that matches criterion.
    :param f: callable
    :param items: list
    :return: int
    """
    for i, x in enumerate(items):
        if f(x):
            return i
    return None


class Context(object):
    """
    A container replacement for global variables to be shared and modified by any function in a module.
    """
    def __init__(self):
        self.ignore = []
        self.ignore.extend(dir(self))

    def update(self, namespace_dict):
        """
        Converts items in a dictionary (such as globals() or locals()) into context object internals.
        :param namespace_dict: dict
        """
        for key, value in namespace_dict.iteritems():
            setattr(self, key, value)

    def __call__(self):
        keys = dir(self)
        for key in self.ignore:
            keys.remove(key)
        return {key: getattr(self, key) for key in keys}


def find_param_value(param_name, x, param_indexes, default_params):
    """

    :param param_name: str
    :param x: arr
    :param param_indexes: dict
    :param default_params: dict
    :return:
    """
    if param_name in param_indexes:
        return float(x[param_indexes[param_name]])
    else:
        return float(default_params[param_name])


def param_array_to_dict(x, param_names):
    """

    :param x: arr
    :param param_names: list
    :return:
    """
    return {param_name: x[ind] for ind, param_name in enumerate(param_names)}


def param_dict_to_array(x_dict, param_names):
    """

    :param x_dict: dict
    :param param_names: list
    :return:
    """
    return np.array([x_dict[param_name] for param_name in param_names])