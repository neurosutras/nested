"""
Library of functions to support nested.parallel
"""
__author__ = 'Aaron D. Milstein and Grace Ng'
try:
    from mpi4py import MPI
except Exception:
    pass
import math
import pickle
import datetime
import copy
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import scipy.optimize
import scipy.signal as signal
import scipy.stats as stats
import random
import pprint
import sys
import os
import gc
import importlib
import traceback
import collections
from collections import Iterable, defaultdict


mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['text.usetex'] = False

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


def nested_convert_scalars(data):
    """
    Crawls a nested dictionary, and converts any scalar objects from numpy types to python types.
    :param data: dict
    :return: dict
    """
    if isinstance(data, dict):
        for key in data:
            data[key] = nested_convert_scalars(data[key])
    elif isinstance(data, Iterable) and not isinstance(data, (str, tuple)):
        for i in range(len(data)):
            data[i] = nested_convert_scalars(data[i])
    elif hasattr(data, 'item'):
        try:
            data = np.asscalar(data)
        except TypeError:
            pass
    return data


def write_to_yaml(file_path, data, convert_scalars=False):
    """

    :param file_path: str (should end in '.yaml')
    :param data: dict
    :param convert_scalars: bool
    :return:
    """
    import yaml
    with open(file_path, 'w') as outfile:
        if convert_scalars:
            data = nested_convert_scalars(data)
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


def null_minimizer(fun, x0, *args, **options):
    """
    Rather than allow scipy.optimize.basinhopping to pass each local mimimum to a gradient descent algorithm for
    polishing, this method catches and passes all local minima so basinhopping can proceed.
    """
    return scipy.optimize.OptimizeResult(x=x0, fun=fun(x0, *args), success=True, nfev=1)


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


def clean_axes(axes, left=True, right=False):
    """
    Remove top and right axes from pyplot axes object.
    :param axes: list of pyplot.Axes
    :param top: bool
    :param left: bool
    :param right: bool
    """
    if not type(axes) in [np.ndarray, list]:
        axes = [axes]
    elif type(axes) == np.ndarray:
        axes = axes.flatten()
    for axis in axes:
        axis.tick_params(direction='out')
        axis.spines['top'].set_visible(False)
        if not right:
            axis.spines['right'].set_visible(False)
        if not left:
            axis.spines['left'].set_visible(False)
        axis.get_xaxis().tick_bottom()
        axis.get_yaxis().tick_left()


def clean_twin_right_axes(axes):
    """
    Remove all but right axis for ax_twin axes.
    :param axes: list of pyplot.Axes
    """
    if not type(axes) in [np.ndarray, list]:
        axes = [axes]
    elif type(axes) == np.ndarray:
        axes = axes.flatten()
    for axis in axes:
        axis.tick_params(direction='out')
        axis.spines['top'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.get_xaxis().tick_bottom()
        axis.get_yaxis().tick_right()


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
    def __init__(self, namespace_dict=None, **kwargs):
        self.update(namespace_dict, **kwargs)

    def update(self, namespace_dict=None, **kwargs):
        """
        Converts items in a dictionary (such as globals() or locals()) into context object internals.
        :param namespace_dict: dict
        """
        if namespace_dict is None:
            namespace_dict = {}
        namespace_dict.update(kwargs)
        for key, value in namespace_dict.iteritems():
            setattr(self, key, value)

    def __call__(self):
        return self.__dict__


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


def print_param_array_like_yaml(param_array, param_names, digits=6):
    """

    :param param_array: dict
    :param param_names: list of str
    :param digits: int
    """
    for ind, param_name in enumerate(param_names):
        param_val = param_array[ind]
        if isinstance(param_val, int):
            print '%s: %s' % (param_name, param_val)
        else:
            print '%s: %.*E' % (param_name, digits, param_val)


def print_param_dict_like_yaml(param_dict, digits=6):
    """

    :param param_dict: dict
    :param digits: int
    """
    for param_name, param_val in param_dict.iteritems():
        if isinstance(param_val, int):
            print '%s: %s' % (param_name, param_val)
        else:
            print '%s: %.*E' % (param_name, digits, param_val)


def get_unknown_click_arg_dict(cli_args):
    """

    :param cli_args: list of str: contains unknown click arguments as list of str
    :return: dict
    """
    kwargs = {}
    for arg in cli_args:
        arg_split = arg.split('=')
        key = arg_split[0][2:]
        if len(arg_split) < 2:
            val = True
        else:
            val = arg_split[1]
        kwargs[key] = val
    return kwargs


# Recursive dictionary merge
# Copyright (C) 2016 Paul Durivage <pauldurivage+github@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.

    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.iteritems():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


def merge_list_of_dict(merge_list):
    """
    Recursive dict merge. Inspired by :meth:``dict.update()``, instead of updating only top-level keys,
    merge_list_of_dict recurses down into dicts nested to an arbitrary depth, updating keys. Each 'merge_dct' in the
    list 'merge_list' is merged, and the resulting dict is returned.
    :param dct: dict onto which the merge is executed
    :param merge_dst: list of dicts to merge into dct
    :return: dict
    """
    dct = dict()
    for merge_dct in merge_list:
        dict_merge(dct, merge_dct)
    return dct


def defaultdict_to_dict(d):
    """

    :param d: nested defaultdict element
    :return: nested defaultdict element
    """
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.iteritems()}
    return d
