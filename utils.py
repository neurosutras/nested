__author__ = 'Aaron D. Milstein and Prannath Moolchand'
"""
Library of functions to support nested.parallel
"""
import sys

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
from collections import defaultdict
from collections.abc import Iterable


mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['text.usetex'] = False

data_dir = 'data/'


def str_to_bool(val):
    """

    :param val: str or int or bool
    :return: bool
    """
    if val in ['true', 'True', '1', 1, True]:
        return True
    elif val in ['false', 'False', '0', 0, False]:
        return False
    else:
        raise Exception('nested.utils.str_to_bool: val: %s with type: %s cannot be interpreted as boolean' %
                        (val, type(val)))


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
        converted_data = dict()
        for key in data:
            converted_key = nested_convert_scalars(key)
            converted_data[converted_key] = nested_convert_scalars(data[key])
        data = converted_data
    elif isinstance(data, Iterable) and not isinstance(data, str):
        data_as_list = list(data)
        for i in range(len(data)):
            data_as_list[i] = nested_convert_scalars(data[i])
        if isinstance(data, tuple):
            data = tuple(data_as_list)
        else:
            data = data_as_list
    elif hasattr(data, 'item'):
        data = data.item()
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
        yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)


def read_from_yaml(file_path, Loader=None):
    """
    Import a python dict from .yaml
    :param file_path: str (should end in '.yaml')
    :param Loader: :class:'yaml.Loader'
    :return: dict
    """
    import yaml
    if Loader is None:
        Loader = yaml.FullLoader
    if os.path.isfile(file_path):
        with open(file_path, 'r') as stream:
            data = yaml.load(stream, Loader=Loader)
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
    indexes = list(range(len(unsorted_x)))
    indexes.sort(key=unsorted_x.__getitem__)
    sorted_x = list(map(unsorted_x.__getitem__, indexes))
    if y is not None:
        sorted_y = list(map(y.__getitem__, indexes))
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
    indexes = list(range(len(str_list)))
    values = []
    for this_str in str_list:
        if end is not None:
            this_str = this_str.split(end)[0]
        this_value = int(this_str.split(seperator)[-1])
        values.append(this_value)
    indexes.sort(key=values.__getitem__)
    sorted_str_list = list(map(str_list.__getitem__, indexes))
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
        if namespace_dict is not None:
            self.__dict__.update(namespace_dict)
        self.__dict__.update(kwargs)

    def __call__(self):
        return self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]


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


def print_param_array_like_yaml(param_array, param_names, digits=None):
    """

    :param param_array: dict
    :param param_names: list of str
    :param digits: int
    """
    for param_name, param_val in zip(param_names, param_array):
        if isinstance(param_val, int) or digits is None:
            print('%s: %s' % (param_name, param_val))
        else:
            print('%s: %.*E' % (param_name, digits, param_val))


def print_param_dict_like_yaml(param_dict, digits=None):
    """

    :param param_dict: dict
    :param digits: int
    """
    for param_name, param_val in param_dict.items():
        if isinstance(param_val, int) or digits is None:
            print('%s: %s' % (param_name, param_val))
        else:
            print('%s: %.*E' % (param_name, digits, param_val))


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
    for k, v in merge_dct.items():
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
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


def get_h5py_group(file, hierarchy, create=False):
    """

    :param file: :class: in ['h5py.File', 'h5py.Group']
    :param hierarchy: list of str
    :param create: bool
    :return: :class:'h5py.Group'
    """
    target = file
    for key in hierarchy:
        if key is not None:
            key = str(key)
            if key not in target:
                if create:
                    target = target.create_group(key)
                else:
                    raise KeyError('get_h5py_group: target: %s does not contain key: %s; valid keys: %s' %
                                   (target, key, list(target.keys())))
            else:
                target = target[key]
    return target


def get_h5py_attr(attrs, key):
    """
    str values are stored as bytes in h5py container attrs dictionaries. This function enables py2/py3 compatibility by
    always returning them to str type upon read. Values should be converted during write with the companion function
    set_h5py_str_attr.
    :param attrs: :class:'h5py._hl.attrs.AttributeManager'
    :param key: str
    :return: val with type converted if str or array of str
    """
    if key not in attrs:
        raise KeyError('get_h5py_attr: invalid key: %s' % key)
    val = attrs[key]
    if isinstance(val, (str, bytes)):
        val = np.string_(val).astype(str)
    elif isinstance(val, Iterable) and len(val) > 0:
        if isinstance(val[0], (str, bytes)):
            val = np.array(val, dtype='str')
    return val


def set_h5py_attr(attrs, key, val):
    """
    str values are stored as bytes in h5py container attrs dictionaries. This function enables py2/py3 compatibility by
    always converting them to np.string_ upon write. Values should be converted back to str during read with the
    companion function get_h5py_str_attr.
    :param attrs: :class:'h5py._hl.attrs.AttributeManager'
    :param key: str
    :param val: type converted if str or array of str
    """
    if isinstance(val, (str, bytes)):
        val = np.string_(val)
    elif isinstance(val, Iterable) and len(val) > 0:
        if isinstance(val[0], (str, bytes)):
            val = np.array(val, dtype='S')
    attrs[key] = val


def nan2None(attr):
    """
    Convert from numpy nan to Python None.
    :param attr: any
    :return: any
    """
    if np.isnan(attr):
        return None
    else:
        return attr


def None2nan(attr):
    """
    Convert from Python None to numpy nan.
    :param attr: any
    :return: any
    """
    if attr is None:
        return np.nan
    else:
        return attr
