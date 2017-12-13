__author__ = 'milsteina'
from mpi4py import MPI
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


#---------------------------------------Some global variables and functions------------------------------

data_dir = 'data/'
morph_dir = 'morphologies/'

freq = 100      # Hz, frequency at which AC length constant will be computed
d_lambda = 0.1  # no segment will be longer than this fraction of the AC length constant

"""
Structure of Mechanism Dictionary: dict of dicts

keys:               description:
'mechanism name':   Value is dictionary specifying how to set parameters at the mechanism level.
'cable':            Value is dictionary specifying how to set basic cable parameters at the section level. Includes
                        'Ra', 'cm', and the special parameter 'spatial_res', which scales the number of segments per
                        section for the specified sec_type by a factor of an exponent of 3.
'ions':             Value is dictionary specifying how to set parameters for ions at the section or segment level.
                    These parameters must be specified **after** all other mechanisms have been inserted.
values:
None:               Use default values for all parameters within this mechanism.
dict:
    keys:
    'parameter name':
    values:     dict:
                        keys:        value:
                        'origin':   'self':     Use 'value' as a baseline value.
                                    sec_type:   Inherit value from last seg of the closest node with sec of
                                                sec_type along the path to root.
                        'value':    float:      If 'origin' is 'self', contains the baseline value.
                        'slope':    float:      If exists, contains slope in units per um. If not, use
                                                constant 'value' for the full length of sec.
                        'max':      float:      If 'slope' exists, 'max' is an upper limit for the value
                        'min':      float:      If 'slope' exists, min is a lower limit for the value

"""

default_mech_dict = {'ais': {'cable': {'Ra': {'origin': 'soma'}, 'cm': {'origin': 'soma'}},
                             'pas': {'e': {'origin': 'soma'}, 'g': {'origin': 'soma'}}},
                     'apical': {'cable': {'Ra': {'origin': 'soma'}, 'cm': {'origin': 'soma'}},
                                'pas': {'e': {'origin': 'soma'}, 'g': {'origin': 'soma'}}},
                     'axon': {'cable': {'Ra': {'origin': 'soma'}, 'cm': {'origin': 'soma'}},
                              'pas': {'e': {'origin': 'soma'}, 'g': {'origin': 'soma'}}},
                     'axon_hill': {'cable': {'Ra': {'origin': 'soma'}, 'cm': {'origin': 'soma'}},
                              'pas': {'e': {'origin': 'soma'}, 'g': {'origin': 'soma'}}},
                     'basal': {'cable': {'Ra': {'origin': 'soma'}, 'cm': {'origin': 'soma'}},
                               'pas': {'e': {'origin': 'soma'}, 'g': {'origin': 'soma'}}},
                     'soma': {'cable': {'Ra': {'value': 150.}, 'cm': {'value': 1.}},
                              'pas': {'e': {'value': -67.}, 'g': {'value': 2.5e-05}}},
                     'trunk': {'cable': {'Ra': {'origin': 'soma'}, 'cm': {'origin': 'soma'}},
                               'pas': {'e': {'origin': 'soma'}, 'g': {'origin': 'soma'}}},
                     'tuft': {'cable': {'Ra': {'origin': 'soma'}, 'cm': {'origin': 'soma'}},
                              'pas': {'e': {'origin': 'soma'}, 'g': {'origin': 'soma'}}},
                     'spine_neck': {'cable': {'Ra': {'origin': 'soma'}, 'cm': {'origin': 'soma'}},
                              'pas': {'e': {'origin': 'soma'}, 'g': {'origin': 'soma'}}},
                     'spine_head': {'cable': {'Ra': {'origin': 'soma'}, 'cm': {'origin': 'soma'}},
                              'pas': {'e': {'origin': 'soma'}, 'g': {'origin': 'soma'}}}}


def lambda_f(sec, f=freq):
    """
    Calculates the AC length constant for the given section at the frequency f
    Used to determine the number of segments per hoc section to achieve the desired spatial and temporal resolution
    :param sec : :class:'h.Section'
    :param f : int
    :return : int
    """
    diam = np.mean([seg.diam for seg in sec])
    Ra = sec.Ra
    cm = np.mean([seg.cm for seg in sec])
    return 1e5*math.sqrt(diam/(4.*math.pi*f*Ra*cm))


def d_lambda_nseg(sec, lam=d_lambda, f=freq):
    """
    The AC length constant for this section and the user-defined fraction is used to determine the maximum size of each
    segment to achieve the d esired spatial and temporal resolution. This method returns the number of segments to set
    the nseg parameter for this section. For tapered cylindrical sections, the diam parameter will need to be
    reinitialized after nseg changes.
    :param sec : :class:'h.Section'
    :param lam : int
    :param f : int
    :return : int
    """
    L = sec.L
    return int((L/(lam*lambda_f(sec, f))+0.9)/2)*2+1


def scaleSWC(filenameBase, mag=100, scope='neurolucida'):
    # this function rescales the SWC file with the real distances.
    f = open(morph_dir+filenameBase+'.swc')
    lines = f.readlines()
    f.close()
    Points = []
    if mag == 100:
        if scope == 'neurolucida':
            xyDist = 0.036909375  # 0.07381875
            zDist = 1.0
        else:
            xyDist = 0.065
            zDist = 0.05
    else:
        raise Exception('Calibration for {}X objective unknown.'.format(mag))
    for line in lines:
        ll = line.split(' ')
        nn = int(float(ll[0]))    # label of the point
        tp = int(float(ll[1]))  # point type
        py = float(ll[2])    # note the inversion of x, y.
        px = float(ll[3])
        z = float(ll[4])    # z
        r = float(ll[5])    # radius of the sphere.
        np = int(float(ll[6]))    # parent point id.
        # get the length in micron
        py *= xyDist; px *= xyDist; r = r*xyDist; z *= zDist
        Points.append([nn,tp,py,px,z,r,np])

    print 'Saving SWC to file '+filenameBase+'-scaled.swc'
    f = open(morph_dir+filenameBase+'-scaled.swc', 'w')
    for [nn,tp,py,px,z,r,np] in Points:
        ll = str(int(nn))+' '+str(int(tp))+' '+str(py)+' '+str(px)+' '+str(z)+' '+str(r)+' '+str(int(np))+'\n'
        f.write(ll)
    f.close()


def investigateSWC(filenameBase):
    # this function reports the min and max values for y, x, z, and radius from an SWC file.
    f = open(morph_dir+filenameBase+'.swc')
    lines = f.readlines()
    f.close()
    xvals = []
    yvals = []
    zvals = []
    rvals = []
    for line in lines:
        ll = line.split(' ')
        yvals.append(float(ll[2]))    # note the inversion of x, y.
        xvals.append(float(ll[3]))
        zvals.append(float(ll[4]))    # z
        rvals.append(float(ll[5]))    # radius of the sphere.
    print 'x - ',min(xvals),':',max(xvals)
    print 'y - ',min(yvals),':',max(yvals)
    print 'z - ',min(zvals),':',max(zvals)
    print 'r - ',min(rvals),':',max(rvals)


def translateSWCs():
    """
    Eric Bloss has produced high resolution .swc files that each contain a volume 10 um deep in z. This method
    determines from the filename the z offset of each file and translates the z coordinates of the .swc files to
    facilitate stitching them together into a single volume. Also changes the sec_type of any node that is not a root
    and has no children within a file to 7 to indicate a leaf that potentially needs to be connected to a nearby root.
    Also attempts to connect unconnected nodes that are within 2 um of each other across consecutive slices, and labels
    them with sec_type = 8. This doesn't work particularly well and files must be extensively proofread in NeuTuMac.
    """
    num_nodes = 0
    outputname = 'combined-offset-connected.swc'
    out_f = open(outputname, 'w')
    # out_test = open('combined-offset-connected.swc', 'w')
    prev_nodes = {}
    filenames = []
    z_offsets = []
    for filename in os.listdir('.'):
        if '.swc' in filename and not '-offset' in filename:
            filenames.append(filename)
            z_offsets.append(float(filename.split('z=')[1].split(' ')[0])/10.0)
    indexes = range(len(z_offsets))
    indexes.sort(key=z_offsets.__getitem__)
    for i in indexes:
        f = open(filenames[i])
        lines = f.readlines()
        f.close()
        num_nodes += len(prev_nodes)
        nodes = {}
        leaves = []
        for line in [line.split(' ') for line in lines if not line.split(' ')[0] in ['#', '\r\n']]:
            index = int(float(line[0])) + num_nodes  # node index
            nodes[index] = {}
            nodes[index]['type'] = int(float(line[1]))  # sec_type
            nodes[index]['y'] = float(line[2])  # note the inversion of x, y.
            nodes[index]['x'] = float(line[3])
            nodes[index]['z'] = float(line[4]) + z_offsets[i]
            nodes[index]['r'] = float(line[5])  # radius of the sphere.
            nodes[index]['parent'] = int(float(line[6]))  # index of parent node
            if not nodes[index]['parent'] == -1:
                nodes[index]['parent'] += num_nodes
                leaves.append(index)
        for index in nodes:  # keep nodes with no children
            parent = nodes[index]['parent']
            if parent in leaves:
                leaves.remove(parent)
        for index in leaves:
            nodes[index]['type'] = 7
        print 'Saving '+filenames[i]+' to '+outputname
        if prev_nodes:
            leaves = [index for index in nodes if (nodes[index]['type'] == 7 or nodes[index]['parent'] == -1)]
            for prev_index in [index for index in prev_nodes if (prev_nodes[index]['type'] == 7 or
                                                                prev_nodes[index]['parent'] == -1)]:
                for index in leaves:
                    distance = math.sqrt((prev_nodes[prev_index]['x']-nodes[index]['x'])**2 +
                                         (prev_nodes[prev_index]['y']-nodes[index]['y'])**2 +
                                         (prev_nodes[prev_index]['z']-nodes[index]['z'])**2)
                    # print prev_index, index, distance
                    if distance < 2.:
                        prev_nodes[prev_index]['type'] = 8
                        nodes[index]['type'] = 8
                        nodes[index]['parent'] = prev_index
                        leaves.remove(index)
                        break
        for index in prev_nodes:
            line = str(index)+' '+str(prev_nodes[index]['type'])+' '+str(prev_nodes[index]['y'])+' '+\
                   str(prev_nodes[index]['x'])+' '+str(prev_nodes[index]['z'])+' '+str(prev_nodes[index]['r'])+' '+\
                   str(prev_nodes[index]['parent'])+'\n'
            out_f.write(line)
        prev_nodes = copy.deepcopy(nodes)
    for index in prev_nodes:
        line = str(index)+' '+str(prev_nodes[index]['type'])+' '+str(prev_nodes[index]['y'])+' '+\
               str(prev_nodes[index]['x'])+' '+str(prev_nodes[index]['z'])+' '+str(prev_nodes[index]['r'])+' '+\
               str(prev_nodes[index]['parent'])+'\n'
        out_f.write(line)
    out_f.close()


def write_to_pkl(fname, data):
    """
    HocCell objects maintain a nested dictionary specifying membrane mechanism parameters for each subcellular
    compartment. This method is used to save that dictionary to a .pkl file that can be read in during model
    specification or after parameter optimization.
    :param fname: str
    :param data: picklable object
    """
    output = open(fname, 'wb')
    pickle.dump(data, output, 2)
    output.close()


def read_from_pkl(fname):
    """
    HocCell objects maintain a nested dictionary specifying membrane mechanism parameters for each subcellular
    compartment. This method is used to load that dictionary from a .pkl file during model specification.
    :param fname: str
    :return: unpickled object
    """
    if os.path.isfile(fname):
        pkl_file = open(fname, 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()
        return data
    else:
        raise Exception('File: {} does not exist.'.format(fname))


def write_to_yaml(file_path, dict):
    """

    :param file_path: str (should end in '.yaml')
    :param dict: dict
    :return:
    """
    import yaml
    with open(file_path, 'w') as outfile:
        yaml.dump(dict, outfile, default_flow_style=False)


def read_from_yaml(file_path):
    """

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


class CheckBounds(object):
    """

    """

    def __init__(self, xmin, xmax):
        """

        :param xmin: dict of float
        :param xmax: dict of float
        """
        self.xmin = xmin
        self.xmax = xmax

    def within_bounds(self, x, param_name):
        """
        For optimize_polish, based on simplex algorithm, check that the current set of parameters are within the bounds.
        :param x: array
        :param param_name: str
        :return: bool
        """
        for i in range(len(x)):
            if ((self.xmin[param_name][i] is not None and x[i] < self.xmin[param_name][i]) or
                    (self.xmax[param_name][i] is not None and x[i] > self.xmax[param_name][i])):
                return False
        return True


class Normalized_Step(object):
    """
    For use with scipy.optimize packages like basinhopping that allow a customized step-taking method.
    Converts basinhopping absolute stepsize into different stepsizes for each parameter such that the stepsizes are
    some fraction of the ranges specified by xmin and xmax. Also enforces bounds for x, and explores the range in
    log10 space when the range is greater than 2 orders of magnitude.
    xmin and xmax are delivered as raw, not relative values. Can handle negative values and ranges that cross zero. If
    xmin and xmax are not provided, or contain None as values, the default is 0.1 and 10. * x0.
    """
    def __init__(self, x0, xmin=None, xmax=None, stepsize=0.5):
        self.stepsize = stepsize
        if xmin is None:
            xmin = [None for i in range(len(x0))]
        if xmax is None:
            xmax = [None for i in range(len(x0))]
        for i in range(len(x0)):
            if xmin[i] is None:
                if x0[i] > 0.:
                    xmin[i] = 0.1 * x0[i]
                else:
                    xmin[i] = 10. * x0[i]
            if xmax[i] is None:
                if x0[i] > 0.:
                    xmax[i] = 10. * x0[i]
                else:
                    xmax[i] = 0.1 * x0[i]
        self.x0 = x0
        self.x_range = np.subtract(xmax, xmin)
        self.order_mag = np.ones_like(x0)
        if not np.any(np.array(xmin) == 0.):
            self.order_mag = np.abs(np.log10(np.abs(np.divide(xmax, xmin))))
        else:
            for i in range(len(x0)):
                if xmin[i] == 0.:
                    self.order_mag[i] = int(xmax[i] / 10)
                else:
                    self.order_mag[i] = abs(np.log10(abs(xmax[i] / xmin[i])))
        self.log10_range = np.log10(np.add(1., self.x_range))
        self.x_offset = np.subtract(1., xmin)

    def __call__(self, current_x):
        x = np.add(current_x, self.x_offset)
        x = np.maximum(x, 1.)
        x = np.minimum(x, np.add(1., self.x_range))
        for i in range(len(x)):
            if self.order_mag[i] >= 2.:
                x[i] = self.log10_step(i, x[i])
            else:
                x[i] = self.linear_step(i, x[i])
        new_x = np.subtract(x, self.x_offset)
        return new_x

    def linear_step(self, i, xi):
        step = self.stepsize * self.x_range[i] / 2.
        new_xi = np.random.uniform(max(1., xi-step), min(xi+step, 1.+self.x_range[i]))
        return new_xi

    def log10_step(self, i, xi):
        step = self.stepsize * self.log10_range[i] / 2.
        xi = np.log10(xi)
        new_xi = np.random.uniform(max(0., xi-step), min(xi+step, self.log10_range[i]))
        new_xi = np.power(10., new_xi)
        return new_xi


def combine_output_files(rec_file_list, new_rec_filename=None, local_data_dir=data_dir):
    """
    List contains names of files generated by "embarassingly parallel" execution of simulations on separate cores.
    This function combines the contents of the files into one .hdf5 file.
    :param rec_file_list: list
    :param new_rec_filename: str or None
    :param local_data_dir: str
    """
    if new_rec_filename is None:
        new_rec_filename = 'combined_output_'+datetime.datetime.today().strftime('%m%d%Y%H%M')
    new_f = h5py.File(local_data_dir+new_rec_filename+'.hdf5', 'w')
    simiter = 0
    for rec_filename in rec_file_list:
        old_f = h5py.File(local_data_dir+rec_filename+'.hdf5', 'r')
        for old_group in old_f.itervalues():
            new_f.copy(old_group, new_f, name=str(simiter))
            simiter += 1
        old_f.close()
    new_f.close()
    print 'Combined data in list of files and exported to: '+new_rec_filename+'.hdf5'
    return new_rec_filename


def combine_hdf5_file_paths(file_path_list, new_file_path=None):
    """
    List contains names of files generated by "embarassingly parallel" execution of simulations on separate cores.
    This function combines the contents of the files into one .hdf5 file.
    :param file_path_list: list of str (paths)
    :param new_file_path: str (path)
    """
    if new_file_path is None:
        raise ValueError('combine_output_file_paths: invalid file path provided: %s' % new_file_path)
    new_f = h5py.File(new_file_path, 'w')
    iter = 0
    for old_file_path in file_path_list:
        old_f = h5py.File(old_file_path, 'r')
        for old_group in old_f.itervalues():
            new_f.copy(old_group, new_f, name=str(iter))
            iter += 1
        old_f.close()
    new_f.close()
    print 'combine_output_file_paths: exported to file path: %s' % new_file_path


def time2index(tvec, start, stop):
    """
    When using adaptive time step (cvode), indices corresponding to specific time points cannot be calculated from a
    fixed dt. This method returns the indices closest to the duration bounded by the specified time points.
    :param tvec: :class:'numpy.array'
    :param start: float
    :param stop: float
    :return: tuple of int
    """
    left = np.where(tvec >= start)[0]
    if np.any(left):  # at least one value was found
        left = left[0]
    else:
        right = len(tvec) - 1  # just take the last two indices
        left = right - 1
        return left, right
    if tvec[left] >= stop:
        right = left
        left -= 1
        return left, right
    right = np.where(tvec <= stop)[0][-1]
    if right == left:
        left -= 1
    return left, right

def interpolate_tvec_vec(tvec, vec, duration, dt=0.02):
    """
    Interpolates the array for tvec from t=0 to t=duration according to the dt time step, and interpolates the
    vec array to correlate with the interpolated tvec array.
    :param tvec: vector of times
    :param vec: vector of voltage recordings
    :param duration: length of time of voltage trace
    :param dt:
    :return:
    """
    interp_t = np.arange(0., duration, dt)
    interp_vm = np.interp(interp_t, tvec, vec)
    return interp_t, interp_vm

def get_Rinp(tvec, vec, start, stop, amp, dt=0.02):
    """
    Calculate peak and steady-state input resistance from a step current injection. For waveform current injections, the
    peak but not the steady-state will have meaning.
    :param tvec: array
    :param vec: array
    :param start: float
    :param stop: float
    :param amp: float
    :param dt: float
    :return: tuple of float
    """

    interp_t, interp_vm = interpolate_tvec_vec(tvec, vec, stop, dt)
    left = int((start-3.) / dt)
    right = left + int(2. / dt)
    baseline = np.mean(interp_vm[left:right])
    temp_vec = np.abs(interp_vm - baseline)
    start_index = int(start / dt)
    peak = np.max(temp_vec[start_index:])
    left = int((stop-3.) / dt)
    right = left + int(2. / dt)
    plateau = np.mean(temp_vec[left:right])
    return baseline, peak/abs(amp), plateau/abs(amp)


def model_exp_rise_decay(t, tau_rise, tau_decay):
    shape = np.exp(-t/tau_decay)-np.exp(-t/tau_rise)
    return shape/np.max(shape)


def model_exp_rise(t, tau):
    return 1-np.exp(-t/tau)


def model_exp_decay(t, tau):
    return np.exp(-t/tau)


def model_scaled_exp(t, A, tau, A0=0):
    return A*np.exp(t/tau)+A0


def null_minimizer(fun, x0, *args, **options):
    """
    Rather than allow basinhopping to pass each local mimimum to a gradient descent algorithm for polishing, this method
    just catches and passes all local minima so basinhopping can proceed.
    """
    return optimize.OptimizeResult(x=x0, fun=fun(x0, *args), success=True, nfev=1)


class MyTakeStep(object):
    """
    For use with scipy.optimize packages like basinhopping that allow a customized step-taking method.
    Converts basinhopping absolute stepsize into different stepsizes for each parameter such that the stepsizes are
    some fraction of the ranges specified by xmin and xmax. Also enforces bounds for x, and explores the range in
    log space when the range is greater than 3 orders of magnitude.
    """
    def __init__(self, blocksize, xmin, xmax, stepsize=0.5):
        self.stepsize = stepsize
        self.blocksize = blocksize
        self.xmin = xmin
        self.xmax = xmax
        self.xrange = []
        for i in range(len(self.xmin)):
            self.xrange.append(self.xmax[i] - self.xmin[i])

    def __call__(self, x):
        for i in range(len(x)):
            if x[i] < self.xmin[i]:
                x[i] = self.xmin[i]
            if x[i] > self.xmax[i]:
                x[i] = self.xmax[i]
            snew = self.stepsize / 0.5 * self.blocksize * self.xrange[i] / 2.
            sinc = min(self.xmax[i] - x[i], snew)
            sdec = min(x[i]-self.xmin[i], snew)
            #  chooses new value in log space to allow fair sampling across orders of magnitude
            if np.log10(self.xmax[i]) - np.log10(self.xmin[i]) >= 3.:
                x[i] = np.power(10, np.random.uniform(np.log10(x[i]-sdec), np.log10(x[i]+sinc)))
            else:
                x[i] += np.random.uniform(-sdec, sinc)
        return x


def get_expected_spine_index_map(sim_file):
    """
    There is a bug with HDF5 when reading from a file too often within a session. Instead of constantly reading from the
    HDF5 file directly and searching for content by spine_index or path_index, the number of calls to the sim_file can
    be reduced by creating a mapping from spine_index or path_index to HDF5 group key. It is possible for a spine to
    have more than one entry in an expected_file, with branch recordings in different locations and therefore different
    expected EPSP waveforms, so it is necessary to also distinguish those entries by path_index.

    :param sim_file: :class:'h5py.File'
    :return: dict
    """
    index_map = {}
    for key, sim in sim_file.iteritems():
        path_index = sim.attrs['path_index']
        spine_index = sim.attrs['spine_index']
        if path_index not in index_map:
            index_map[path_index] = {}
        index_map[path_index][spine_index] = key
    return index_map


def get_spine_group_info(sim_filename, verbose=1):
    """
    Given a processed output file generated by export_nmdar_cooperativity, this method returns a dict that has
    separated each group of stimulated spines by dendritic sec_type, and sorted by distance from soma. For ease of
    inspection so that the appropriate path_index can be chosen for plotting expected and actual summation traces.
    :param sim_filename: str
    :return: dict
    """
    spine_group_info = {}
    with h5py.File(data_dir+sim_filename+'.hdf5', 'r') as f:
        for path_index in f:
            sim = f[path_index]
            path_type = sim.attrs['path_type']
            path_category = sim.attrs['path_category']
            if path_type not in spine_group_info:
                spine_group_info[path_type] = {}
            if path_category not in spine_group_info[path_type]:
                spine_group_info[path_type][path_category] = {'path_indexes': [], 'distances': []}
            if path_index not in spine_group_info[path_type][path_category]['path_indexes']:
                spine_group_info[path_type][path_category]['path_indexes'].append(path_index)
                if path_type == 'apical':
                    # for obliques, sort by the distance of the branch origin from the soma
                    distance = sim.attrs['origin_distance']
                else:
                    distance = sim.attrs['soma_distance']
                spine_group_info[path_type][path_category]['distances'].append(distance)
    for path_type in spine_group_info:
        for path_category in spine_group_info[path_type]:
            indexes = range(len(spine_group_info[path_type][path_category]['distances']))
            indexes.sort(key=spine_group_info[path_type][path_category]['distances'].__getitem__)
            spine_group_info[path_type][path_category]['distances'] = \
                map(spine_group_info[path_type][path_category]['distances'].__getitem__, indexes)
            spine_group_info[path_type][path_category]['path_indexes'] = \
                map(spine_group_info[path_type][path_category]['path_indexes'].__getitem__, indexes)
        if verbose:
            for path_category in spine_group_info[path_type]:
                print path_type, '-', path_category
                for i, distance in enumerate(spine_group_info[path_type][path_category]['distances']):
                    print spine_group_info[path_type][path_category]['path_indexes'][i], distance
    return spine_group_info


def get_expected_EPSP(sim_file, group_index, equilibrate, duration, dt=0.02):
    """
    Given an output file generated by parallel_clustered_branch_cooperativity or build_expected_EPSP_reference, this
    method returns a dict of numpy arrays, each containing the depolarization-rectified expected EPSP for each
    recording site resulting from stimulating a single spine.
    :param sim_file: :class:'h5py.File'
    :param group_index: int
    :param equilibrate: float
    :param duration: float
    :param dt: float
    :return: dict of :class:'numpy.array'
    """
    sim = sim_file[str(group_index)]
    t = sim['time'][:]
    interp_t = np.arange(0., duration, dt)
    left, right = time2index(interp_t, equilibrate-3., equilibrate-1.)
    start, stop = time2index(interp_t, equilibrate-2., duration)
    trace_dict = {}
    for rec in sim['rec'].itervalues():
        location = rec.attrs['description']
        vm = rec[:]
        interp_vm = np.interp(interp_t, t, vm)
        baseline = np.average(interp_vm[left:right])
        interp_vm -= baseline
        interp_vm = interp_vm[start:stop]
        """
        rectified = np.zeros(len(interp_vm))
        rectified[np.where(interp_vm>0.)[0]] += interp_vm[np.where(interp_vm>0.)[0]]
        trace_dict[location] = rectified
        """
        peak = np.max(interp_vm)
        peak_index = np.where(interp_vm == peak)[0][0]
        zero_index = np.where(interp_vm[peak_index:] <= 0.)[0]
        if np.any(zero_index):
            interp_vm[peak_index+zero_index[0]:] = 0.
        trace_dict[location] = interp_vm
    interp_t = interp_t[start:stop]
    interp_t -= interp_t[0] + 2.
    trace_dict['time'] = interp_t
    return trace_dict


def get_expected_vs_actual(expected_sim_file, actual_sim_file, expected_index_map, sorted_actual_sim_keys,
                           interval=0.3, dt=0.02):
    """
    Given an output file generated by parallel_clustered_branch_cooperativity, and an output file generated by
    parallel_branch_cooperativity, this method returns a dict of lists, each containing an input-output function
    relating expected to actual peak depolarization for each recording site from stimulating a group of spines on a
    single branch or path. The variable expected_index_map contains a dictionary that converts an integer spine_index to
    a string group_index to locate the expected EPSP for a given spine in the expected_sim_file. The variable
    sorted_actual_sim_keys contains the indexes of the simulations in the actual_sim_file corresponding to the branch or
    path, ordered by number of stimulated spines. These variables must be pre-computed.
    :param expected_sim_file: :class:'h5py.File'
    :param actual_sim_file: :class:'h5py.File'
    :param expected_index_map: dict
    :param sorted_actual_sim_keys: list of str
    :param interval: float
    :return: dict of list
    """
    equilibrate = actual_sim_file[sorted_actual_sim_keys[0]].attrs['equilibrate']
    duration = actual_sim_file[sorted_actual_sim_keys[0]].attrs['duration']
    actual = {}
    for sim in [actual_sim_file[key] for key in sorted_actual_sim_keys]:
        t = sim['time'][:]
        interp_t = np.arange(0., duration, dt)
        left, right = time2index(interp_t, equilibrate-3., equilibrate-1.)
        start, stop = time2index(interp_t, equilibrate-2., duration)
        for rec in sim['rec'].itervalues():
            location = rec.attrs['description']
            if not location in actual:
                actual[location] = []
            vm = rec[:]
            interp_vm = np.interp(interp_t, t, vm)
            baseline = np.average(interp_vm[left:right])
            interp_vm -= baseline
            interp_vm = interp_vm[start:stop]
            actual[location].append(np.max(interp_vm))
    spine_list = sim.attrs['syn_indexes']
    interp_t = interp_t[start:stop]
    interp_t -= interp_t[0] + 2.
    expected = {}
    summed_traces = {}
    equilibrate = expected_sim_file.itervalues().next().attrs['equilibrate']
    duration = expected_sim_file.itervalues().next().attrs['duration']
    for i, spine_index in enumerate(spine_list):
        group_index = expected_index_map[spine_index]
        trace_dict = get_expected_EPSP(expected_sim_file, group_index, equilibrate, duration, dt)
        t = trace_dict['time']
        left, right = time2index(interp_t, -2.+i*interval, interp_t[-1])
        right = min(right, left+len(t))
        for location in [location for location in trace_dict if not location == 'time']:
            trace = trace_dict[location]
            if not location in expected:
                expected[location] = []
                summed_traces[location] = np.zeros(len(interp_t))
            summed_traces[location][left:right] += trace[:right-left]
            expected[location].append(np.max(summed_traces[location]))
    return expected, actual


def export_nmdar_cooperativity(expected_filename, actual_filename, description="", output_filename=None):
    """
    Expects expected and actual files to be generated by parallel_clustered_ or
    parallel_distributed_branch_cooperativity. Files contain simultaneous voltage recordings from 3 locations (soma,
    trunk, dendrite origin) during synchronous stimulation of branches, and an NMDAR conductance recording from a single
    spine in each group. Spines are distributed across 4 dendritic sec_types (basal, trunk, apical, tuft).
    Generates a processed output file containing expected vs. actual data and metadata for each group of spines.
    Can be used to generate plots of supralinearity, NMDAR conductance, or average across conditions, etc.
    :param expected_filename: str
    :param actual_filename: str
    :param description: str
    :param output_filename: str
    """
    sim_key_dict = {}
    with h5py.File(data_dir+actual_filename+'.hdf5', 'r') as actual_file:
        for key, sim in actual_file.iteritems():
            path_index = sim.attrs['path_index']
            if path_index not in sim_key_dict:
                sim_key_dict[path_index] = []
            sim_key_dict[path_index].append(key)
        with h5py.File(data_dir+expected_filename+'.hdf5', 'r') as expected_file:
            expected_index_map = get_expected_spine_index_map(expected_file)
            with h5py.File(data_dir+output_filename+'.hdf5', 'w') as output_file:
                output_file.attrs['description'] = description
                for path_index in sim_key_dict:
                    path_group = output_file.create_group(str(path_index))
                    sim_keys = sim_key_dict[path_index]
                    sim_keys.sort(key=lambda x: len(actual_file[x].attrs['syn_indexes']))
                    sim = actual_file[sim_keys[0]]
                    path_type = sim.attrs['path_type']
                    path_category = sim.attrs['path_category']
                    soma_distance = sim['rec']['4'].attrs['soma_distance']
                    branch_distance = sim['rec']['4'].attrs['branch_distance']
                    origin_distance = soma_distance - branch_distance
                    path_group.attrs['path_type'] = path_type
                    path_group.attrs['path_category'] = path_category
                    path_group.attrs['soma_distance'] = soma_distance
                    path_group.attrs['branch_distance'] = branch_distance
                    path_group.attrs['origin_distance'] = origin_distance
                    expected_dict, actual_dict = get_expected_vs_actual(expected_file, actual_file,
                                                                        expected_index_map[path_index], sim_keys)
                    for rec in sim['rec'].itervalues():
                        location = rec.attrs['description']
                        rec_group = path_group.create_group(location)
                        rec_group.create_dataset('expected', compression='gzip', compression_opts=9,
                                                           data=expected_dict[location])
                        rec_group.create_dataset('actual', compression='gzip', compression_opts=9,
                                                           data=actual_dict[location])


def get_instantaneous_spike_probability(rate, dt=0.02, generator=None):
    """
    Given an instantaneous spike rate in Hz and a time bin in ms, calculate whether a spike has occurred in that time
    bin, assuming an exponential distribution of inter-spike intervals.
    :param rate: float (Hz)
    :param dt: float (ms)
    :param generator: :class:'random.Random'
    :return: bool
    """
    if generator is None:
        generator = random
    x = generator.uniform(0., 1.)
    rate /= 1000.
    p = 1. - np.exp(-rate * dt)
    return bool(x < p)


def get_inhom_poisson_spike_times(rate, t, dt=0.02, refractory=3., generator=None):
    """
    Given a time series of instantaneous spike rates in Hz, produce a spike train consistent with an inhomogeneous
    Poisson process with a refractory period after each spike.
    :param rate: instantaneous rates in time (Hz)
    :param t: corresponding time values (ms)
    :param dt: temporal resolution for spike times (ms)
    :param refractory: absolute deadtime following a spike (ms)
    :param generator: :class:'random.Random()'
    :return: list of m spike times (ms)
    """
    if generator is None:
        generator = random
    interp_t = np.arange(t[0], t[-1]+dt, dt)
    interp_rate = np.interp(interp_t, t, rate)
    spike_times = []
    i = 0
    while i < len(interp_t):
        if get_instantaneous_spike_probability(interp_rate[i], dt, generator):
            spike_times.append(interp_t[i])
            i += int(refractory / dt)
        else:
            i += 1
    return spike_times


def get_inhom_poisson_spike_times_by_thinning(rate, t, dt=0.02, refractory=3., generator=None):
    """
    Given a time series of instantaneous spike rates in Hz, produce a spike train consistent with an inhomogeneous
    Poisson process with a refractory period after each spike.
    :param rate: instantaneous rates in time (Hz)
    :param t: corresponding time values (ms)
    :param dt: temporal resolution for spike times (ms)
    :param refractory: absolute deadtime following a spike (ms)
    :param generator: :class:'random.Random()'
    :return: list of m spike times (ms)
    """
    if generator is None:
        generator = random
    interp_t = np.arange(t[0], t[-1] + dt, dt)
    interp_rate = np.interp(interp_t, t, rate)
    interp_rate /= 1000.
    non_zero = np.where(interp_rate > 0.)[0]
    interp_rate[non_zero] = 1. / (1. / interp_rate[non_zero] - refractory)
    spike_times = []
    max_rate = np.max(interp_rate)
    i = 0
    ISI_memory = 0.
    while i < len(interp_t):
        x = generator.random()
        if x > 0.:
            ISI = -np.log(x) / max_rate
            i += int(ISI / dt)
            ISI_memory += ISI
            if (i < len(interp_t)) and (generator.random() <= interp_rate[i] / max_rate) and ISI_memory >= 0.:
                spike_times.append(interp_t[i])
                ISI_memory = -refractory
    return spike_times


def get_binned_firing_rate(train, t, bin_dur=10.):
    """

    :param train: array of float
    :param t: array
    :param bin_dur: float (ms)
    :return: array (Hz)
    """
    bin_centers = np.arange(t[0]+bin_dur/2., t[-1], bin_dur)
    count = np.zeros(len(bin_centers))
    for spike_time in train:
        if t[0] <= spike_time <= bin_centers[-1] + bin_dur / 2.:
            i = np.where(bin_centers + bin_dur / 2. >= spike_time)[0][0]
            count[i] += 1
    rate = count / bin_dur * 1000.
    rate = np.interp(t, bin_centers, rate)
    return rate


def get_smoothed_firing_rate(train, t, bin_dur=50., bin_step=10., dt=0.1):
    """

    :param train: array of float
    :param t: array
    :param bin_dur: float (ms)
    :param dt: float (ms)
    :return: array (Hz)
    """
    count = np.zeros(len(t))
    bin_centers = np.arange(t[0]+bin_dur/2., t[-1]-bin_dur/2., bin_step)
    rate = np.zeros(len(bin_centers))
    for spike_time in train:
        if spike_time >= t[0] and spike_time <= t[-1]:
            i = np.where(t >= spike_time)[0][0]
            count[i] += 1
    left = 0
    right = int(bin_dur / dt)
    interval = int(bin_step / dt)
    for i, bin_t in enumerate(bin_centers):
        rate[i] = np.sum(count[left:right]) / bin_dur * 1000.
        left += interval
        right += interval
    rate = np.interp(t, bin_centers, rate)
    return rate


def get_removed_spikes(rec_filename, before=1.6, after=6., dt=0.02, th=10., plot=1):
    """

    :param rec_filename: str
    :param before: float : time to remove before spike
    :param after: float : time to remove after spike in case where trough or voltage recovery cannot be used
    :param dt: float : temporal resolution for interpolation and dvdt
    :param th: float : slope threshold
    :param plot: int
    :return: list of :class:'numpy.array'
    """
    removed = []
    with h5py.File(data_dir+rec_filename+'.hdf5', 'r') as f:
        sim = f.itervalues().next()
        equilibrate = sim.attrs['equilibrate']
        duration = sim.attrs['duration']
        track_equilibrate = sim.attrs['track_equilibrate']
        for rec in f.itervalues():
            t = np.arange(0., duration, dt)
            vm = np.interp(t, rec['time'], rec['rec']['0'])
            start = int((equilibrate + track_equilibrate) / dt)
            t = np.subtract(t[start:], equilibrate + track_equilibrate)
            vm = vm[start:]
            dvdt = np.gradient(vm, dt)
            crossings = np.where(dvdt >= th)[0]
            if not np.any(crossings):
                removed.append(vm)
            else:
                start = 0.
                i = 0
                left = max(0, crossings[i] - int(before/dt))
                previous_vm = vm[left]
                while i < len(crossings) and start < len(vm):
                    start = crossings[i]
                    left = max(0, crossings[i] - int(before/dt))
                    if not np.isnan(vm[left]):
                        previous_vm = vm[left]
                    recovers = np.where(vm[crossings[i]:] < previous_vm)[0]
                    if np.any(recovers):
                        recovers = crossings[i] + recovers[0]
                    falling = np.where(dvdt[crossings[i]:] < 0.)[0]
                    if np.any(falling):
                        falling = crossings[i] + falling[0]
                        rising = np.where(dvdt[falling:] >= 0.)[0]
                        if np.any(rising):
                            rising = falling + rising[0]
                    else:
                        rising = []
                    if np.any(recovers):
                        if np.any(rising):
                            right = min(recovers, rising)
                        else:
                            right = recovers
                    elif np.any(rising):
                        right = rising
                    else:
                        right = min(crossings[i] + int(after/dt), len(vm)-1)
                    # added to remove majority of complex spike:
                    if vm[right] >= -45. and np.any(recovers):
                        right = recovers
                    for j in range(left, right):
                        vm[j] = np.nan
                    i += 1
                    while i < len(crossings) and crossings[i] < right:
                        i += 1
                not_blank = np.where(~np.isnan(vm))[0]
                vm = np.interp(t, t[not_blank], vm[not_blank])
                removed.append(vm)
            temp_t = np.arange(0., duration, dt)
            temp_vm = np.interp(temp_t, rec['time'], rec['rec']['0'])
            start = int((equilibrate + track_equilibrate) / dt)
            temp_t = np.subtract(temp_t[start:], equilibrate + track_equilibrate)
            temp_vm = temp_vm[start:]
            if plot:
                plt.plot(temp_t, temp_vm)
                plt.plot(t, vm)
                plt.show()
                plt.close()
    return removed


def get_removed_spikes_alt(rec_filename, before=1.6, after=6., dt=0.02, th=10., plot=1, rec_key='0'):
    """

    :param rec_filename: str
    :param before: float : time to remove before spike
    :param after: float : time to remove after spike in case where trough or voltage recovery cannot be used
    :param dt: float : temporal resolution for interpolation and dvdt
    :param th: float : slope threshold
    :param plot: int
    :param rec_key: str
    :return: list of :class:'numpy.array'
    """
    removed = []
    with h5py.File(data_dir+rec_filename+'.hdf5', 'r') as f:
        sim = f.itervalues().next()
        equilibrate = sim.attrs['equilibrate']
        duration = sim.attrs['duration']
        track_equilibrate = sim.attrs['track_equilibrate']
        for trial in f.itervalues():
            t = np.arange(0., duration, dt)
            vm = np.interp(t, trial['time'], trial['rec'][rec_key])
            start = int((equilibrate + track_equilibrate) / dt)
            t = np.subtract(t[start:], equilibrate + track_equilibrate)
            vm = vm[start:]
            dvdt = np.gradient(vm, dt)
            crossings = np.where(dvdt >= th)[0]
            if not np.any(crossings):
                removed.append(vm)
            else:
                start = 0.
                i = 0
                left = max(0, crossings[i] - int(before/dt))
                previous_vm = vm[left]
                while i < len(crossings) and start < len(vm):
                    start = crossings[i]
                    left = max(0, crossings[i] - int(before/dt))
                    right = min(crossings[i] + int(after/dt), len(vm) - 1)
                    if not np.isnan(vm[left]):
                        previous_vm = vm[left]
                    recovers = np.where(vm[crossings[i]:] < previous_vm)[0]
                    if np.any(recovers):
                        recovers = crossings[i] + recovers[0]
                    falling = np.where(dvdt[crossings[i]:] < 0.)[0]
                    if np.any(falling):
                        falling = crossings[i] + falling[0]
                        rising = np.where(dvdt[falling:] >= 0.)[0]
                        if np.any(rising):
                            rising = falling + rising[0]
                    else:
                        rising = []
                    if np.any(rising):
                        right = min(right, rising)
                    # added to remove majority of complex spike:
                    if vm[right] >= -35. and np.any(recovers):
                        right = recovers
                    for j in range(left, right):
                        vm[j] = np.nan
                    i += 1
                    while i < len(crossings) and crossings[i] < right:
                        i += 1
                not_blank = np.where(~np.isnan(vm))[0]
                vm = np.interp(t, t[not_blank], vm[not_blank])
                removed.append(vm)
            temp_t = np.arange(0., duration, dt)
            temp_vm = np.interp(temp_t, trial['time'], trial['rec'][rec_key])
            start = int((equilibrate + track_equilibrate) / dt)
            temp_t = np.subtract(temp_t[start:], equilibrate + track_equilibrate)
            temp_vm = temp_vm[start:]
            if plot:
                plt.plot(temp_t, temp_vm)
                plt.plot(t, vm)
                plt.show()
                plt.close()
    return removed


def remove_spikes_from_array(vm, t, before=1.6, after=6., dt=0.02, th=10., plot=False):
    """

    :param vm: array
    :param t: array
    :param before: float : time to remove before spike
    :param after: float : time to remove after spike in case where trough or voltage recovery cannot be used
    :param dt: float : temporal resolution for interpolation and dvdt
    :param th: float : slope threshold
    :param plot: bool
    :return: array
    """
    temp_vm = np.array(vm)
    dvdt = np.gradient(vm, dt)
    crossings = np.where(dvdt >= th)[0]
    if not np.any(crossings):
        removed = np.array(vm)
    else:
        start = 0.
        i = 0
        left = max(0, crossings[i] - int(before / dt))
        previous_vm = temp_vm[left]
        while i < len(crossings) and start < len(temp_vm):
            start = crossings[i]
            left = max(0, crossings[i] - int(before / dt))
            if not np.isnan(temp_vm[left]):
                previous_vm = temp_vm[left]
            recovers = np.where(temp_vm[crossings[i]:] < previous_vm)[0]
            if np.any(recovers):
                recovers = crossings[i] + recovers[0]
            falling = np.where(dvdt[crossings[i]:] < 0.)[0]
            if np.any(falling):
                falling = crossings[i] + falling[0]
                rising = np.where(dvdt[falling:] >= 0.)[0]
                if np.any(rising):
                    rising = falling + rising[0]
            else:
                rising = []
            if np.any(recovers):
                if np.any(rising):
                    right = min(recovers, rising)
                else:
                    right = recovers
            elif np.any(rising):
                right = rising
            else:
                right = min(crossings[i] + int(after/dt), len(temp_vm)-1)
            # added to remove majority of complex spike:
            if temp_vm[right] >= -45. and np.any(recovers):
                right = recovers
            for j in range(left, right):
                temp_vm[j] = np.nan
            i += 1
            while i < len(crossings) and crossings[i] < right:
                i += 1
        not_blank = np.where(~np.isnan(temp_vm))[0]
        removed = np.interp(t, t[not_blank], temp_vm[not_blank])
    if plot:
        plt.plot(t, vm)
        plt.plot(t, removed)
        plt.show()
        plt.close()
    return removed


def get_removed_spikes_nangaps(rec_filename, before=1.6, after=6., dt=0.02, th=10., rec_key='0'):
    """
    Return traces with spikes removed and interpolated, but also traces with spikes replaced with nan to keep track of
    where spike removal occurred.
    :param rec_filename: str
    :param before: float : time to remove before spike
    :param after: float : time to remove after spike in case where trough or voltage recovery cannot be used
    :param dt: float : temporal resolution for interpolation and dvdt
    :param th: float : slope threshold
    :param rec_key: str : default is '0', which is the somatic recording
    :return: list of :class:'numpy.array', list of :class:'numpy.array'
    """
    removed_interp = []
    removed_nangaps = []
    with h5py.File(data_dir+rec_filename+'.hdf5', 'r') as f:
        sim = f.itervalues().next()
        equilibrate = sim.attrs['equilibrate']
        duration = sim.attrs['duration']
        track_equilibrate = sim.attrs['track_equilibrate']
        for trial in f.itervalues():
            t = np.arange(0., duration, dt)
            vm = np.interp(t, trial['time'], trial['rec'][rec_key])
            start = int((equilibrate + track_equilibrate) / dt)
            t = np.subtract(t[start:], equilibrate + track_equilibrate)
            vm = vm[start:]
            dvdt = np.gradient(vm, dt)
            crossings = np.where(dvdt >= th)[0]
            if not np.any(crossings):
                removed_interp.append(vm)
            else:
                start = 0.
                i = 0
                left = max(0, crossings[i] - int(before/dt))
                previous_vm = vm[left]
                while i < len(crossings) and start < len(vm):
                    start = crossings[i]
                    left = max(0, crossings[i] - int(before/dt))
                    if not np.isnan(vm[left]):
                        previous_vm = vm[left]
                    recovers = np.where(vm[crossings[i]:] < previous_vm)[0]
                    if np.any(recovers):
                        recovers = crossings[i] + recovers[0]
                    falling = np.where(dvdt[crossings[i]:] < 0.)[0]
                    if np.any(falling):
                        falling = crossings[i] + falling[0]
                        rising = np.where(dvdt[falling:] >= 0.)[0]
                        if np.any(rising):
                            rising = falling + rising[0]
                    else:
                        rising = []
                    if np.any(recovers):
                        if np.any(rising):
                            right = min(recovers, rising)
                        else:
                            right = recovers
                    elif np.any(rising):
                        right = rising
                    else:
                        right = min(crossings[i] + int(after/dt), len(vm)-1)
                    # added to remove majority of complex spike:
                    if vm[right] >= -45. and np.any(recovers):
                        right = recovers
                    for j in range(left, right):
                        vm[j] = np.nan
                    i += 1
                    while i < len(crossings) and crossings[i] < right:
                        i += 1
                not_blank = np.where(~np.isnan(vm))[0]
                removed_nangaps.append(vm)
                vm = np.interp(t, t[not_blank], vm[not_blank])
                removed_interp.append(vm)
    return removed_interp, removed_nangaps


def get_theta_filtered_traces(rec_filename, dt=0.02):
    """

    :param rec_file_name: str
    # remember .attrs['phase_offset'] could be inside ['train'] for old files
    """
    with h5py.File(data_dir+rec_filename+'.hdf5', 'r') as f:
        sim = f.itervalues().next()
        equilibrate = sim.attrs['equilibrate']
        track_equilibrate = sim.attrs['track_equilibrate']
        track_length = sim.attrs['track_length']
        input_field_duration = sim.attrs['input_field_duration']
        duration = sim.attrs['duration']
        stim_dt = sim.attrs['stim_dt']
        track_duration = duration - equilibrate - track_equilibrate
        stim_t = np.arange(-track_equilibrate, track_duration, stim_dt)
        exc_input = []
        inh_input = []
        phase_offsets = []
        for sim in f.itervalues():
            exc_input_sum = None
            inh_input_sum = None
            for train in sim['train'].itervalues():
                this_exc_rate = get_binned_firing_rate(np.array(train), stim_t)
                if exc_input_sum is None:
                    exc_input_sum = np.array(this_exc_rate)
                else:
                    exc_input_sum = np.add(exc_input_sum, this_exc_rate)
            exc_input.append(exc_input_sum)
            for train in sim['inh_train'].itervalues():
                this_inh_rate = get_binned_firing_rate(np.array(train), stim_t)
                if inh_input_sum is None:
                    inh_input_sum = np.array(this_inh_rate)
                else:
                    inh_input_sum = np.add(inh_input_sum, this_inh_rate)
            inh_input.append(inh_input_sum)
            if 'phase_offset' in sim.attrs:
                phase_offsets.append(sim.attrs['phase_offset'])
            elif 'phase_offset' in sim['train'].attrs:
                phase_offsets.append(sim['train'].attrs['phase_offset'])
    rec_t = np.arange(0., track_duration, dt)
    spikes_removed = get_removed_spikes(rec_filename, plot=0)
    # down_sample traces to 2 kHz after clipping spikes for theta and ramp filtering
    down_dt = 0.5
    down_stim_t = np.arange(-track_equilibrate, track_duration, down_dt)
    down_rec_t = np.arange(0., track_duration, down_dt)
    # 2000 ms Hamming window, ~3 Hz low-pass for ramp, ~5 - 10 Hz bandpass for theta
    window_len = min(int(2000./down_dt), len(down_rec_t) - 1)
    theta_filter = signal.firwin(window_len, [5., 10.], nyq=1000./2./down_dt, pass_zero=False)
    pop_exc_theta = []
    pop_inh_theta = []
    intra_theta = []
    for pop in exc_input:
        down_sampled = np.interp(down_stim_t, stim_t, pop)
        filtered = signal.filtfilt(theta_filter, [1.], down_sampled, padtype='even', padlen=window_len)
        up_sampled = np.interp(stim_t, down_stim_t, filtered)
        pop_exc_theta.append(up_sampled)
    for pop in inh_input:
        down_sampled = np.interp(down_stim_t, stim_t, pop)
        filtered = signal.filtfilt(theta_filter, [1.], down_sampled, padtype='even', padlen=window_len)
        up_sampled = np.interp(stim_t, down_stim_t, filtered)
        pop_inh_theta.append(up_sampled)
    for trace in spikes_removed:
        down_sampled = np.interp(down_rec_t, rec_t, trace)
        filtered = signal.filtfilt(theta_filter, [1.], down_sampled, padtype='even', padlen=window_len)
        up_sampled = np.interp(rec_t, down_rec_t, filtered)
        intra_theta.append(up_sampled)
    return stim_t, pop_exc_theta, pop_inh_theta, rec_t, intra_theta, phase_offsets


def get_phase_precession(rec_filename, start_loc=None, end_loc=None, theta_duration=None, dt=0.02):
    """

    :param rec_file_name: str
    :param start_loc: float
    :param end_loc: float
    :param dt: 0.02
    # remember .attrs['phase_offset'] could be inside ['train'] for old files
    """
    with h5py.File(data_dir+rec_filename+'.hdf5', 'r') as f:
        sim = f.itervalues().next()
        equilibrate = sim.attrs['equilibrate']
        track_equilibrate = sim.attrs['track_equilibrate']
        duration = sim.attrs['duration']
        track_duration = duration - equilibrate - track_equilibrate
        if theta_duration is None:
            if 'global_theta_cycle_duration' in sim.attrs:
                theta_duration = sim.attrs['global_theta_cycle_duration']
            else:
                theta_duration = 150.
        if start_loc is None:
            start_loc = 0.
        if end_loc is None:
            end_loc = track_duration
        phase_offsets = []
        for sim in f.itervalues():
            if 'phase_offset' in sim.attrs:
                phase_offsets.append(sim.attrs['phase_offset'])
            elif 'train' in sim and 'phase_offset' in sim['train'].attrs:
                phase_offsets.append(sim['train'].attrs['phase_offset'])
            else:
                phase_offsets.append(0.)
        output_trains = [np.array(sim['output']) for sim in f.itervalues() if 'output' in sim]
    spike_phase_array = []
    spike_time_array = []
    for i, train in enumerate(output_trains):
        time_offset = phase_offsets[i]
        on_track = np.where((train >= start_loc) & (train <= end_loc))[0]
        if not np.any(on_track):
            spike_phase_array.append([])
            spike_time_array.append([])
        else:
            spike_times = train[on_track]
            spike_time_array.append(spike_times)
            spike_times = np.subtract(spike_times, time_offset)
            spike_phases = np.mod(spike_times, theta_duration)
            spike_phases /= theta_duration
            spike_phases *= 360.
            spike_phase_array.append(spike_phases)
    rec_t = np.arange(0., track_duration, dt)
    spikes_removed = get_removed_spikes(rec_filename, plot=0)
    # down_sample traces to 2 kHz after clipping spikes for theta filtering
    down_dt = 0.5
    down_rec_t = np.arange(0., track_duration, down_dt)
    # 2000 ms Hamming window, ~5 - 10 Hz bandpass for theta
    window_len = min(len(down_rec_t) - 1, int(2000. / down_dt))
    pad_len = int(window_len / 2.)
    theta_filter = signal.firwin(window_len, [5., 10.], nyq=1000. / 2. / down_dt, pass_zero=False)
    intra_theta = []
    for trace in spikes_removed:
        down_sampled = np.interp(down_rec_t, rec_t, trace)
        padded_trace = np.zeros(len(down_sampled) + window_len)
        padded_trace[pad_len:-pad_len] = down_sampled
        padded_trace[:pad_len] = down_sampled[::-1][-pad_len:]
        padded_trace[-pad_len:] = down_sampled[::-1][:pad_len]
        filtered = signal.filtfilt(theta_filter, [1.], padded_trace, padlen=pad_len)
        filtered = filtered[pad_len:-pad_len]
        up_sampled = np.interp(rec_t, down_rec_t, filtered)
        intra_theta.append(up_sampled)
    intra_peak_array = []
    intra_phase_array = []
    for i, trial in enumerate(intra_theta):
        time_offset = phase_offsets[i]
        peak_locs = signal.argrelmax(trial)[0]
        peak_times = rec_t[peak_locs]
        intra_peak_array.append(peak_times)
        peak_times = np.subtract(peak_times, time_offset)
        peak_phases = np.mod(peak_times, theta_duration)
        peak_phases /= theta_duration
        peak_phases *= 360.
        intra_phase_array.append(peak_phases)
    return spike_time_array, spike_phase_array, intra_peak_array, intra_phase_array


def get_phase_precession_live(t, vm, spikes=None, time_offset=0., theta_duration = 150., start_loc=None, end_loc=None,
                              dt=0.02, adjust=False):
    """

    :param vm: array
    :param t: array
    :param phase_offset: float
    :param theta_duration: float
    :param start_loc: float
    :param end_loc: float
    :param dt: 0.02
    """
    if start_loc is None:
        start_loc = t[0]
    if end_loc is None:
        end_loc = t[-1]
    if spikes is not None:
        on_track = np.where((spikes >= start_loc) & (spikes <= end_loc))[0]
        if not np.any(on_track):
            spike_phases = []
            spike_times = []
        else:
            spike_times = spikes[on_track]
            spike_times_offset = np.subtract(spike_times, time_offset)
            spike_phases = np.mod(spike_times_offset, theta_duration)
            spike_phases /= theta_duration
            spike_phases *= 360.
    else:
        spike_phases = []
        spike_times = []
    rec_t = np.arange(t[0], t[-1]+dt, dt)
    vm = np.interp(rec_t, t, vm)
    if spikes is not None:
        spikes_removed = remove_spikes_from_array(vm, rec_t, plot=0)
    else:
        spikes_removed = np.array(vm)
    # down_sample traces to 2 kHz after clipping spikes for theta filtering
    down_dt = 0.5
    down_rec_t = np.arange(t[0], t[-1]+down_dt, down_dt)
    # 2000 ms Hamming window, ~5 - 10 Hz bandpass for theta
    window_len = min(len(down_rec_t) - 1, int(2000. / down_dt))
    pad_len = int(window_len / 2.)
    theta_filter = signal.firwin(window_len, [5., 10.], nyq=1000. / 2. / down_dt, pass_zero=False)
    down_sampled = np.interp(down_rec_t, rec_t, spikes_removed)
    padded_trace = np.zeros(len(down_sampled) + window_len)
    padded_trace[pad_len:-pad_len] = down_sampled
    padded_trace[:pad_len] = down_sampled[::-1][-pad_len:]
    padded_trace[-pad_len:] = down_sampled[::-1][:pad_len]
    filtered = signal.filtfilt(theta_filter, [1.], padded_trace, padlen=pad_len)
    filtered = filtered[pad_len:-pad_len]
    intra_theta = np.interp(rec_t, down_rec_t, filtered)
    peak_locs = signal.argrelmax(intra_theta)[0]
    intra_peaks = rec_t[peak_locs]
    intra_peaks_offset = np.subtract(intra_peaks, time_offset)
    intra_phases = np.mod(intra_peaks_offset, theta_duration)
    intra_phases /= theta_duration
    intra_phases *= 360.
    if adjust:
        for i in range(1, len(intra_phases)):
            if (intra_phases[i]-intra_phases[i-1] > 90.) and (intra_phases[i-1] < 180.):
                intra_phases[i] -= 360.
            elif (intra_phases[i] - intra_phases[i - 1]) < -90. and (intra_phases[i - 1] > 180.):
                intra_phases[i] += 360.
    return rec_t, vm, spikes_removed, intra_theta, spike_times, spike_phases, intra_peaks, intra_phases


def get_input_spike_train_phase_precession(rec_filename, index, start_loc=None, end_loc=None, dt=0.02):
    """

    :param rec_file_name: str
    :param index: str: key to particular input train
    # remember .attrs['phase_offset'] could be inside ['train'] for old files
    """
    with h5py.File(data_dir+rec_filename+'.hdf5', 'r') as f:
        sim = f.itervalues().next()
        equilibrate = sim.attrs['equilibrate']
        track_equilibrate = sim.attrs['track_equilibrate']
        duration = sim.attrs['duration']
        stim_dt = sim.attrs['stim_dt']
        track_duration = duration - equilibrate - track_equilibrate
        if 'global_theta_cycle_duration' in sim.attrs:
            theta_duration = sim.attrs['global_theta_cycle_duration']
        else:
            theta_duration = 150.
        if start_loc is None:
            start_loc = 0.
        if end_loc is None:
            end_loc = track_duration
        spike_phase_array = []
        spike_time_array = []
        for sim in f.itervalues():
            if 'phase_offset' in sim.attrs:
                time_offset = sim.attrs['phase_offset']
            elif 'train' in sim and 'phase_offset' in sim['train'].attrs:
                time_offset = sim['train'].attrs['phase_offset']
            else:
                time_offset = 0.
            train = sim['train'][index]
            on_track = np.where((train >= start_loc) & (train <= end_loc))[0]
            if not np.any(on_track):
                spike_phase_array.append([])
                spike_time_array.append([])
            else:
                spike_times = np.array(train)[on_track]
                spike_time_array.append(spike_times)
                spike_times = np.subtract(spike_times, time_offset)
                spike_phases = np.mod(spike_times, theta_duration)
                spike_phases /= theta_duration
                spike_phases *= 360.
                spike_phase_array.append(spike_phases)
    return spike_time_array, spike_phase_array


def get_subset_downsampled_recordings(rec_filename, description, dt=0.1):
    """

    :param rec_file_name: str
    # remember .attrs['phase_offset'] could be inside ['train'] for old files
    """
    with h5py.File(data_dir+rec_filename+'.hdf5', 'r') as f:
        sim = f.itervalues().next()
        equilibrate = sim.attrs['equilibrate']
        track_equilibrate = sim.attrs['track_equilibrate']
        duration = sim.attrs['duration']
        rec_t = np.arange(0., duration, dt)
        sim_list = []
        for sim in f.itervalues():
            rec_list = []
            index_list = []
            for rec in [rec for rec in sim['rec'].itervalues() if 'description' in rec.attrs and
                            rec.attrs['description'] == description]:
                down_sampled = np.interp(rec_t, sim['time'], rec)
                rec_list.append(down_sampled[int((equilibrate + track_equilibrate) / dt):])
                index_list.append(rec.attrs['index'])
            sim_list.append({'index_list': index_list, 'rec_list': rec_list})
    rec_t = np.arange(0., duration - track_equilibrate - equilibrate, dt)
    return rec_t, sim_list


def get_patterned_input_r_inp(rec_filename, seperate=False):
    """
    Expects a simulation file in which an oscillating somatic current injection was used to probe input resistance.
    separate toggle can be used to either separate or combine measurements from the rising and falling phases of the
    current injection.
    :param rec_filename: str
    :param seperate: bool
    :return: hypo_r_inp_array, hypo_phase_array, hypo_t_array, depo_r_inp_array, depo_phase_array, depo_t_array
    """
    with h5py.File(data_dir+rec_filename+'.hdf5', 'r') as f:
        sim = f.itervalues().next()
        equilibrate = sim.attrs['equilibrate']
        track_equilibrate = sim.attrs['track_equilibrate']
        duration = sim.attrs['duration']
        duration -= equilibrate + track_equilibrate
        dt = sim.attrs['stim_dt']
        theta_cycle_duration = sim.attrs['global_theta_cycle_duration']
        probe_amp = sim.attrs['r_inp_probe_amp']
        probe_dur = sim.attrs['r_inp_probe_duration']
        phase_offsets = [trial.attrs['phase_offset'] for trial in f.itervalues()]
        traces = get_removed_spikes(rec_filename, plot=0)
        hypo_r_inp_array, hypo_phase_array, hypo_t_array = [], [], []
        depo_r_inp_array, depo_phase_array, depo_t_array = [], [], []
        for i, vm in enumerate(traces):
            phase_offset = phase_offsets[i]
            start = 0.
            while start + probe_dur < duration:
                r_inp = (vm[int((start + probe_dur) / dt)] - vm[int(start / dt)]) / probe_amp
                phase = ((start + probe_dur / 2. - phase_offset) % theta_cycle_duration) / theta_cycle_duration * 360.
                hypo_r_inp_array.append(r_inp)
                hypo_phase_array.append(phase)
                hypo_t_array.append(start + probe_dur / 2.)
                start += probe_dur
                if start + probe_dur < duration:
                    r_inp = (vm[int((start + probe_dur) / dt)] - vm[int(start / dt)]) / (probe_amp * -1.)
                    phase = ((start + probe_dur / 2. - phase_offset) % theta_cycle_duration) / theta_cycle_duration \
                                                                                                * 360.
                    depo_r_inp_array.append(r_inp)
                    depo_phase_array.append(phase)
                    depo_t_array.append(start + probe_dur / 2.)
                    start += probe_dur
    if seperate:
        return hypo_r_inp_array, hypo_phase_array, hypo_t_array, depo_r_inp_array, depo_phase_array, depo_t_array
    else:
        return np.append(hypo_r_inp_array, depo_r_inp_array), np.append(hypo_phase_array, depo_phase_array), \
               np.append(hypo_t_array, depo_t_array)


def get_patterned_input_component_traces(rec_filename, dt=0.02):
    """

    :param rec_file_name: str
    # remember .attrs['phase_offset'] could be inside ['train'] for old files
    """
    with h5py.File(data_dir+rec_filename+'.hdf5', 'r') as f:
        sim = f.itervalues().next()
        equilibrate = sim.attrs['equilibrate']
        track_equilibrate = sim.attrs['track_equilibrate']
        duration = sim.attrs['duration']
        track_duration = duration - equilibrate - track_equilibrate
        start = int((equilibrate + track_equilibrate)/dt)
        vm_array = []
        for sim in f.itervalues():
            t = np.arange(0., duration, dt)
            vm = np.interp(t, sim['time'], sim['rec']['0'])
            vm = vm[start:]
            vm_array.append(vm)
    rec_t = np.arange(0., track_duration, dt)
    spikes_removed = get_removed_spikes(rec_filename, plot=0, dt=dt)
    # down_sample traces to 2 kHz after clipping spikes for theta and ramp filtering
    down_dt = 0.5
    down_t = np.arange(0., track_duration, down_dt)
    # 2000 ms Hamming window, ~2 Hz low-pass for ramp, ~5 - 10 Hz bandpass for theta
    window_len = int(2000./down_dt)
    pad_len = int(window_len/2.)
    theta_filter = signal.firwin(window_len, [5., 10.], nyq=1000./2./down_dt, pass_zero=False)
    ramp_filter = signal.firwin(window_len, 2., nyq=1000./2./down_dt)
    theta_traces = []
    ramp_traces = []
    for trace in spikes_removed:
        down_sampled = np.interp(down_t, rec_t, trace)
        padded_trace = np.zeros(len(down_sampled)+window_len)
        padded_trace[pad_len:-pad_len] = down_sampled
        padded_trace[:pad_len] = down_sampled[::-1][-pad_len:]
        padded_trace[-pad_len:] = down_sampled[::-1][:pad_len]
        filtered = signal.filtfilt(theta_filter, [1.], padded_trace, padlen=pad_len)
        filtered = filtered[pad_len:-pad_len]
        up_sampled = np.interp(rec_t, down_t, filtered)
        theta_traces.append(up_sampled)
        filtered = signal.filtfilt(ramp_filter, [1.], padded_trace, padlen=pad_len)
        filtered = filtered[pad_len:-pad_len]
        up_sampled = np.interp(rec_t, down_t, filtered)
        ramp_traces.append(up_sampled)
    return rec_t, vm_array, theta_traces, ramp_traces


def alternative_binned_vm_variance_analysis(rec_filename, dt=0.02):
    """
    When removing spikes, produce a linear interpolated trace as well as a trace where spikes are replaced with nan.
    Filter and subtract theta and ramp in the usual way, but replace regions of the residual traces corresponding to
    spike removal with nan for the purpose of calculating variance in spatial bins. Also generate the raw vm
    distributions without the potential artifact of linear spike interpolation.
    :param rec_filename: str
    :param dt: float
    """
    with h5py.File(data_dir + rec_filename + '.hdf5', 'r') as f:
        sim = f.itervalues().next()
        equilibrate = sim.attrs['equilibrate']
        track_equilibrate = sim.attrs['track_equilibrate']
        track_length = sim.attrs['track_length']
        input_field_duration = sim.attrs['input_field_duration']
        duration = sim.attrs['duration']
        track_duration = duration - equilibrate - track_equilibrate
        spatial_bin = input_field_duration / 50.
    rec_t = np.arange(0., track_duration, dt)
    spikes_removed_interp, spikes_removed_nangaps = get_removed_spikes_nangaps(rec_filename, th=10.)
    spikes_removed_interp = get_removed_spikes_alt(rec_filename, plot=0)
    # down_sample traces to 2 kHz after clipping spikes for theta and ramp filtering
    down_dt = 0.5
    down_t = np.arange(0., track_duration, down_dt)
    # 2000 ms Hamming window, ~2 Hz low-pass for ramp, ~5 - 10 Hz bandpass for theta, ~0.2 Hz low-pass for residuals
    window_len = int(2000. / down_dt)
    pad_len = int(window_len / 2.)
    theta_filter = signal.firwin(window_len, [5., 10.], nyq=1000./2./down_dt, pass_zero=False)
    ramp_filter = signal.firwin(window_len, 2., nyq=1000./2./down_dt)
    slow_vm_filter = signal.firwin(window_len, .2, nyq=1000./2./down_dt)
    theta_traces = []
    theta_removed = []
    ramp_traces = []
    residuals_interp = []
    residuals_nangaps = []
    theta_envelopes = []
    for trace in spikes_removed_interp:
        down_sampled = np.interp(down_t, rec_t, trace)
        padded_trace = np.zeros(len(down_sampled) + window_len)
        padded_trace[pad_len:-pad_len] = down_sampled
        padded_trace[:pad_len] = down_sampled[::-1][-pad_len:]
        padded_trace[-pad_len:] = down_sampled[::-1][:pad_len]
        filtered = signal.filtfilt(theta_filter, [1.], padded_trace, padlen=pad_len)
        this_theta_envelope = np.abs(signal.hilbert(filtered))
        filtered = filtered[pad_len:-pad_len]
        up_sampled = np.interp(rec_t, down_t, filtered)
        theta_traces.append(up_sampled)
        this_theta_removed = trace - up_sampled
        theta_removed.append(this_theta_removed)
        this_theta_envelope = this_theta_envelope[pad_len:-pad_len]
        up_sampled = np.interp(rec_t, down_t, this_theta_envelope)
        theta_envelopes.append(up_sampled)
        filtered = signal.filtfilt(ramp_filter, [1.], padded_trace, padlen=pad_len)
        filtered = filtered[pad_len:-pad_len]
        up_sampled = np.interp(rec_t, down_t, filtered)
        ramp_traces.append(up_sampled)
        filtered = signal.filtfilt(slow_vm_filter, [1.], padded_trace, padlen=pad_len)
        filtered = filtered[pad_len:-pad_len]
        up_sampled = np.interp(rec_t, down_t, filtered)
        residual_interp = this_theta_removed - up_sampled
        residuals_interp.append(residual_interp)
    """
    for i, trace in enumerate(spikes_removed_nangaps):
        residual_nangaps = np.array(residuals_interp[i])
        nan_indexes = np.where(np.isnan(trace))[0]
        if np.any(nan_indexes):
            residual_nangaps[nan_indexes] = np.nan
        residuals_nangaps.append(residual_nangaps)
    """
    binned_mean = [[] for i in range(len(residuals_interp))]
    binned_variance = [[] for i in range(len(residuals_interp))]
    binned_t = []
    bin_duration = 3. * spatial_bin
    interval = int(bin_duration / dt)
    for j in range(0, int(track_duration / bin_duration)):
        binned_t.append(j * bin_duration + bin_duration / 2.)
        for i, residual in enumerate(residuals_interp):
            binned_variance[i].append(np.var(residual[j * interval:(j + 1) * interval]))
            #binned_variance[i].append(np.nanvar(residuals_nangaps[i][j * interval:(j + 1) * interval]))
            binned_mean[i].append(np.mean(theta_removed[i][j * interval:(j + 1) * interval]))
    mean_theta_envelope = np.mean(theta_envelopes, axis=0)
    mean_ramp = np.mean(ramp_traces, axis=0)
    mean_binned_vm = np.mean(binned_mean, axis=0)
    mean_binned_var = np.mean(binned_variance, axis=0)
    scatter_vm_mean = np.array(binned_mean).flatten()
    scatter_vm_var = np.array(binned_variance).flatten()
    return rec_t, residuals_interp, mean_theta_envelope, scatter_vm_mean, scatter_vm_var, binned_t, mean_binned_vm, \
           mean_binned_var, mean_ramp


def get_patterned_input_mean_values(residuals, intra_theta_amp, rate_map, ramp, key_list=None,
                                    i_bounds=[0., 1800., 3600., 5400.], peak_bounds=[600., 1200., 4200., 4800.],
                                    dt=0.02):
    """

    :param residuals: dict of np.array
    :param intra_theta_amp: dict of np.array
    :param rate_map: dict of np.array
    :param ramp: dict of np.array
    :param key_list: list of str
    :param i_bounds: list of float, time points corresponding to inhibitory manipulation for variance measurement
    :param peak_bounds: list of float, time points corresponding to 10 "spatial bins" for averaging
    :param dt: float, temporal resolution
    """
    if key_list is None:
        key_list = ['modinh0', 'modinh1', 'modinh2']
    baseline = np.mean(ramp[key_list[0]][:int(600. / dt)])
    key_list.extend([key_list[0]+'_out', key_list[0]+'_in'])
    mean_var, mean_theta_amp, mean_rate, mean_ramp = {}, {}, {}, {}
    for source_condition, target_condition in zip([key_list[1], key_list[0]], [key_list[1], key_list[3]]):
        start = int(i_bounds[0]/dt)
        end = int(i_bounds[1]/dt)
        mean_var[target_condition] = np.mean([np.nanvar(residual[start:end]) for residual in
                                              residuals[source_condition]])
        start = int(peak_bounds[0]/dt)
        end = int(peak_bounds[1]/dt)
        mean_theta_amp[target_condition] = np.mean(intra_theta_amp[source_condition][start:end])
        mean_rate[target_condition] = np.mean(rate_map[source_condition][start:end])
        mean_ramp[target_condition] = np.mean(ramp[source_condition][start:end]) - baseline
    for source_condition, target_condition in zip([key_list[2], key_list[0]], [key_list[2], key_list[4]]):
        start = int(i_bounds[2]/dt)
        end = int(i_bounds[3]/dt)
        mean_var[target_condition] = np.mean([np.nanvar(residual[start:end]) for residual in
                                              residuals[source_condition]])
        start = int(peak_bounds[2]/dt)
        end = int(peak_bounds[3]/dt)
        mean_theta_amp[target_condition] = np.mean(intra_theta_amp[source_condition][start:end])
        mean_rate[target_condition] = np.mean(rate_map[source_condition][start:end])
        mean_ramp[target_condition] = np.mean(ramp[source_condition][start:end]) - baseline
    for parameter, title in zip([mean_var, mean_theta_amp, mean_rate, mean_ramp],
                                ['Variance: ', 'Theta Envelope: ', 'Rate: ', 'Depolarization: ']):
        print title
        for condition in key_list[1:]:
            print condition, parameter[condition]


def get_i_syn_mean_values(parameter_array, parameter_title, key_list=None, peak_bounds=[600., 1200., 4200., 4800.],
                          dt=0.02):
    """

    :param parameter_array: dict of np.array
    :param parameter_title: str: meant to be in ['i_AMPA', 'i_NMDA', 'i_GABA', 'E:I Ratio']
    :param key_list: list of str
    :param peak_bounds: list of float, time points corresponding to 10 "spatial bins" for averaging
    :param dt: float, temporal resolution
    """
    if key_list is None:
        key_list = ['modinh0', 'modinh1', 'modinh2']
    key_list.extend([key_list[0] + '_out', key_list[0] + '_in'])
    mean_val = {}
    print parameter_title+':'
    for source_condition, target_condition in zip([key_list[1], key_list[0]], [key_list[1], key_list[3]]):
        start = int(peak_bounds[0] / dt)
        end = int(peak_bounds[1] / dt)
        mean_val[target_condition] = np.mean(parameter_array[source_condition][start:end])
        print target_condition+': ', mean_val[target_condition]
    for source_condition, target_condition in zip([key_list[2], key_list[0]], [key_list[2], key_list[4]]):
        start = int(peak_bounds[2] / dt)
        end = int(peak_bounds[3] / dt)
        mean_val[target_condition] = np.mean(parameter_array[source_condition][start:end])
        print target_condition + ': ', mean_val[target_condition]


def compress_i_syn_rec_files(rec_filelist, rec_description_list=['i_AMPA', 'i_NMDA', 'i_GABA'],
                             local_data_dir=data_dir):
    """
    Simulations in which synaptic currents have been recorded from all active synapses produce very large files, but
    only the total sum current is analyzed. This function replaces the individual current recordings in each .hdf5
    output file with single summed current traces for each synapse type (e.g. i_AMPA, i_NMDA, i_GABA) to reduce
    file sizes. Can be run on the storage server before local transfer and analysis.
    :param rec_filelist: list of str
    :param rec_description_list: list of str
    :param local_data_dir: str
    """
    for rec_file in rec_filelist:
        with h5py.File(local_data_dir+rec_file+'.hdf5', 'a') as f:
            for trial in f.itervalues():
                group_dict = {}
                for rec in trial['rec']:
                    key = trial['rec'][rec].attrs['description']
                    if key in rec_description_list:
                        if key not in group_dict:
                            group_dict[key] = np.array(trial['rec'][rec])
                        else:
                            group_dict[key] = np.add(group_dict[key], trial['rec'][rec])
                        del trial['rec'][rec]
                for key in group_dict:
                    trial.create_dataset(key, compression='gzip', compression_opts=9, data=group_dict[key])
        print 'Compressed group recordings in file: ', rec_file


def process_i_syn_rec(rec_filename, description_list=['i_AMPA', 'i_NMDA', 'i_GABA'], dt=0.02):
    """
    Expects a simulation file generated by test_poisson_inputs_record_i_syn, which has been compressed by the function
    compress_i_syn_rec_files, and now contains a summed current recording. This method returns the averaged waveform
    across trials, as well as the average of a low-pass filtered waveform across trials.
    :param rec_filename: str
    :param description_list: list of str
    :param dt: float
    :return: dict
    """
    with h5py.File(data_dir+rec_filename+'.hdf5', 'r') as f:
        sim = f.itervalues().next()
        equilibrate = sim.attrs['equilibrate']
        track_equilibrate = sim.attrs['track_equilibrate']
        duration = sim.attrs['duration']
        track_duration = duration - equilibrate - track_equilibrate
        t = np.arange(0., duration, dt)
        rec_t = np.arange(0., track_duration, dt)
        start = int((equilibrate + track_equilibrate) / dt)
        group_dict = {}
        for trial in f.itervalues():
            for key in trial:
                if key in description_list:
                    if key not in group_dict:
                        group_dict[key] = []
                    group = np.interp(t, trial['time'], trial[key])
                    group_dict[key].append(group[start:])
        group_mean_dict = {}
        for key in group_dict:
            group_mean_dict[key] = np.mean(group_dict[key], axis=0)
        down_dt = 0.5
        down_t = np.arange(0., track_duration, down_dt)
        # 2000 ms Hamming window, ~3 Hz low-pass filter
        window_len = int(2000./down_dt)
        pad_len = int(window_len / 2.)
        ramp_filter = signal.firwin(window_len, 2., nyq=1000. / 2. / down_dt)
        group_low_pass_dict = {key: [] for key in group_dict}
        for key in group_dict:
            for group in group_dict[key]:
                down_sampled = np.interp(down_t, rec_t, group)
                padded_trace = np.zeros(len(down_sampled) + window_len)
                padded_trace[pad_len:-pad_len] = down_sampled
                padded_trace[:pad_len] = down_sampled[::-1][-pad_len:]
                padded_trace[-pad_len:] = down_sampled[::-1][:pad_len]
                filtered = signal.filtfilt(ramp_filter, [1.], padded_trace, padlen=pad_len)
                filtered = filtered[pad_len:-pad_len]
                up_sampled = np.interp(rec_t, down_t, filtered)
                group_low_pass_dict[key].append(up_sampled)
        group_mean_low_pass_dict = {}
        for key in group_low_pass_dict:
            group_mean_low_pass_dict[key] = np.mean(group_low_pass_dict[key], axis=0)
        return rec_t, group_mean_dict, group_mean_low_pass_dict


def process_special_rec_within_group(rec_filename, group_name='pre',
                                     description_list=['soma', 'proximal_trunk', 'distal_trunk'], dt=0.02):
    """

    :param rec_filename: str
    :param group_name: str
    :param description_list: list of str
    :param dt: float
    :return: dict
    """
    with h5py.File(data_dir+rec_filename+'.hdf5', 'r') as f:
        sim = f.itervalues().next()
        equilibrate = sim.attrs['equilibrate']
        track_equilibrate = sim.attrs['track_equilibrate']
        duration = sim.attrs['duration']
        track_duration = duration - equilibrate - track_equilibrate
        t = np.arange(0., duration, dt)
        rec_t = np.arange(0., track_duration, dt)
        start = int((equilibrate + track_equilibrate) / dt)
        rec_dict = {}
        for trial in f.itervalues():
            for rec in trial[group_name].itervalues():
                description = rec.attrs['description']
                if description in description_list:
                    if description not in rec_dict:
                        rec_dict[description] = []
                    this_rec = np.interp(t, trial['time'], rec)
                    rec_dict[description].append(this_rec[start:])
        rec_mean_dict = {}
        for description in rec_dict:
            rec_mean_dict[description] = np.mean(rec_dict[description], axis=0)
        down_dt = 0.5
        down_t = np.arange(0., track_duration, down_dt)
        # 2000 ms Hamming window, ~3 Hz low-pass filter
        window_len = int(2000./down_dt)
        print 'This method hasn\'t been updated with appropriate signal padding before filtering.'
        ramp_filter = signal.firwin(window_len, 2., nyq=1000./2./down_dt)
        rec_low_pass_dict = {description: [] for description in rec_dict}
        for description in rec_dict:
            for rec in rec_dict[description]:
                down_sampled = np.interp(down_t, rec_t, rec)
                filtered = signal.filtfilt(ramp_filter, [1.], down_sampled, padtype='even', padlen=window_len)
                up_sampled = np.interp(rec_t, down_t, filtered)
                rec_low_pass_dict[description].append(up_sampled)
        rec_mean_low_pass_dict = {}
        for description in rec_low_pass_dict:
            rec_mean_low_pass_dict[description] = np.mean(rec_low_pass_dict[description], axis=0)
        return rec_t, rec_mean_dict, rec_mean_low_pass_dict


def generate_patterned_input_expected(expected_filename, actual_filename, output_filename=None,
                                      location_list=['soma'], dt=0.02, P0=0.2, local_random=None):
    """
    Given a reference file containing unitary EPSPs corresponding to stimulating individual spines in isolation, and an
    actual file from a patterned input simulation, generate an output file containing two sets of expected waveforms: a
    purely post-synaptic expectation based on linear summation of successful release events, and a combined pre- and
    post-synaptic expectation assuming linear summation of the expected value of transmission, including a low basal P0.
    :param expected_filename: str
    :param actual_filename: str
    :param output_filename: str
    :param location_list: list of str
    :param dt: float
    :param P0: float
    :param local_random: :class:'random.Random'
    :return: str: output_filename
    """
    if local_random is None:
        local_random = random
    if output_filename is None:
        output_filename = actual_filename+'_linear_expected'
    with h5py.File(data_dir+expected_filename+'.hdf5', 'r') as expected_file:
        expected_equilibrate = expected_file.itervalues().next().attrs['equilibrate']
        expected_duration = expected_file.itervalues().next().attrs['duration']
        expected_key_map = {expected_file[key].attrs['spine_index']: key for key in expected_file}
        with h5py.File(data_dir+actual_filename+'.hdf5', 'r') as actual_file:
            trial = actual_file.itervalues().next()
            equilibrate = trial.attrs['equilibrate']
            track_equilibrate = trial.attrs['track_equilibrate']
            duration = trial.attrs['duration']
            track_duration = duration - track_equilibrate - equilibrate
            t = np.arange(0., track_equilibrate + track_duration, dt)
            stochastic = True if 'successes' in trial else False
            expected_sim = expected_file.itervalues().next()
            expected_t = np.arange(0., expected_duration, dt)
            baseline = {}
            for rec in (rec for rec in expected_sim['rec'].itervalues() if rec.attrs['description'] in location_list):
                sec_type = rec.attrs['description']
                vm = np.interp(expected_t, expected_sim['time'], rec)
                baseline[sec_type] = np.mean(vm[int((expected_equilibrate-3.)/dt):int((expected_equilibrate-1.)/dt)])
            for trial_index in actual_file:
                pre = {sec_type: np.zeros(len(t))+baseline[sec_type] for sec_type in location_list}
                post = {sec_type: np.zeros(len(t))+baseline[sec_type] for sec_type in location_list}
                trial = actual_file[trial_index]
                for train_index in trial['train']:
                    synapse_index = trial['train'][train_index].attrs['index']
                    expected_EPSP = get_expected_EPSP(expected_file, expected_key_map[synapse_index],
                                                      expected_equilibrate, expected_duration, dt)
                    for spike_time in trial['train'][train_index]:
                        start = int((spike_time + track_equilibrate)/dt)
                        this_success = local_random.uniform(0., 1.) < P0
                        if not stochastic or this_success:
                            for sec_type in location_list:
                                if sec_type in expected_EPSP:
                                    this_EPSP = expected_EPSP[sec_type][int(2./dt):]
                                    stop = min(start + len(this_EPSP), len(t))
                                    if stochastic:
                                        pre[sec_type][start:stop] += this_EPSP[:stop-start]
                                    else:
                                        post[sec_type][start:stop] += this_EPSP[:stop-start]
                    if stochastic:
                        for spike_time in trial['successes'][train_index]:
                            start = int((spike_time + track_equilibrate)/dt)
                            for sec_type in location_list:
                                if sec_type in expected_EPSP:
                                    this_EPSP = expected_EPSP[sec_type][int(2./dt):]
                                    stop = min(start + len(this_EPSP), len(t))
                                    post[sec_type][start:stop] += this_EPSP[:stop-start]
                with h5py.File(data_dir+output_filename+'.hdf5', 'a') as output_file:
                    output_file.create_group(trial_index)
                    output_file[trial_index].attrs['dt'] = dt
                    output_file[trial_index].attrs['equilibrate'] = 0.
                    output_file[trial_index].attrs['track_equilibrate'] = track_equilibrate
                    output_file[trial_index].attrs['duration'] = track_equilibrate + track_duration
                    output_file[trial_index].create_dataset('time', compression='gzip', compression_opts=9, data=t)
                    if stochastic:
                        output_file[trial_index].create_group('pre')
                        for i, sec_type in enumerate(location_list):
                            output_file[trial_index]['pre'].create_dataset(str(i), compression='gzip',
                                                                           compression_opts=9, data=pre[sec_type])
                            output_file[trial_index]['pre'][str(i)].attrs['description'] = sec_type
                    output_file[trial_index].create_group('post')
                    for i, sec_type in enumerate(location_list):
                        output_file[trial_index]['post'].create_dataset(str(i), compression='gzip',
                                                                       compression_opts=9, data=post[sec_type])
                        output_file[trial_index]['post'][str(i)].attrs['description'] = sec_type
    return output_filename


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


def fit_power_func(p, x):
    """
    Expects [y0, x0, exponent] to fit an exponential relationship with an unknown power.
    :param p: array
    :param x: array
    :return: array
    """
    return p[0] + p[1] * ((x - p[2]) ** p[3])


def error_power_func(p, x, y):
    """
    Expects [y0, x0, exponent] to fit an exponential relationship with an unknown power. Evaluates error relative to y.
    :param p:
    :param x:
    :param y:
    :return: float
    """
    return np.sum((y - fit_power_func(p, x)) ** 2.)


def get_waveform_phase_vs_time(t, x=None, cycle_duration=150., time_offset=0.):
    """
    Given either a spike train (t alone), or a timecourse and waveform, remove a phase offset and report the times and
    phases of either spikes or peaks.
    :param t: array
    :param x: array
    :param cycle_duration: float
    :param time_offset: float
    :return: list of array
    """
    if x is not None:
        peak_locs = signal.argrelmax(x)[0]
        peak_times = t[peak_locs]
    else:
        peak_times = t
    peak_phases = np.mod(np.subtract(peak_times, time_offset), cycle_duration)
    peak_phases /= cycle_duration
    peak_phases *= 360.
    return peak_times, peak_phases


def low_pass_filter(source, freq, duration, dt, down_dt=0.5):
    """
    Filters the source waveform at the provided frequency.
    :param source: array
    :param freq: float
    :param duration: float
    :param dt: float
    :param down_dt: float
    :return: np.array: filtered source
    """
    t = np.arange(0., duration+dt, dt)
    t = t[:len(source)]
    down_t = np.arange(0., duration, down_dt)
    # 2000 ms Hamming window
    window_len = int(2000. / down_dt)
    pad_len = int(window_len / 2.)
    lp_filter = signal.firwin(window_len, freq, nyq=1000. / 2. / down_dt)
    down_sampled = np.interp(down_t, t, source)
    padded_trace = np.zeros(len(down_sampled) + window_len)
    padded_trace[pad_len:-pad_len] = down_sampled
    padded_trace[:pad_len] = down_sampled[::-1][-pad_len:]
    padded_trace[-pad_len:] = down_sampled[::-1][:pad_len]
    filtered = signal.filtfilt(lp_filter, [1.], padded_trace, padlen=pad_len)
    filtered = filtered[pad_len:-pad_len]
    up_sampled = np.interp(t, down_t, filtered)[:len(source)]
    return up_sampled


def general_filter_trace(t, source, filter, duration, dt, down_dt=0.5):
    """
    Filters the source waveform at the provided frequency.
    :param t: array
    :param source: array
    :param filter: 'signal.firwin'
    :param duration: float
    :param dt: float
    :param down_dt: float
    :return: np.array: filtered source
    """
    down_t = np.arange(0., duration, down_dt)
    # 2000 ms Hamming window
    window_len = int(2000. / down_dt)
    pad_len = int(window_len / 2.)
    down_sampled = np.interp(down_t, t, source)
    padded_trace = np.zeros(len(down_sampled) + window_len)
    padded_trace[pad_len:-pad_len] = down_sampled
    padded_trace[:pad_len] = down_sampled[::-1][-pad_len:]
    padded_trace[-pad_len:] = down_sampled[::-1][:pad_len]
    filtered = signal.filtfilt(filter, [1.], padded_trace, padlen=pad_len)
    filtered = filtered[pad_len:-pad_len]
    up_sampled = np.interp(t, down_t, filtered)
    return up_sampled


def print_ramp_features(x, ramp, title, track_length=None, dx=None, induction_loc=None, plot=False):
    """

    :param x: array
    :param ramp: array
    :param title: str
    :param dx = None
    :param induction_loc: float
    :param plot: bool
    """
    if track_length is None:
        track_length = 187.
    if induction_loc is None:
        induction_loc = track_length/2.
    binned_x = np.array(x)
    if dx is None:
        dx = 1. * 30. / 1000.
    default_interp_x = np.arange(0., track_length, dx)
    extended_binned_x = np.concatenate([binned_x - track_length, binned_x, binned_x + track_length])
    extended_binned_ramp = np.concatenate([ramp for i in range(3)])
    extended_interp_x = np.concatenate([default_interp_x - track_length, default_interp_x,
                                        default_interp_x + track_length])
    dx = extended_interp_x[1] - extended_interp_x[0]
    extended_ramp = np.interp(extended_interp_x, extended_binned_x, extended_binned_ramp)
    interp_ramp = extended_ramp[len(default_interp_x):2 * len(default_interp_x)]
    baseline_indexes = np.where(interp_ramp <= np.percentile(interp_ramp, 10.))[0]
    baseline = np.mean(interp_ramp[baseline_indexes])
    interp_ramp -= baseline
    extended_ramp -= baseline
    peak_index = np.where(interp_ramp == np.max(interp_ramp))[0][0] + len(interp_ramp)
    # use center of mass in 10 spatial bins instead of literal peak for determining peak_shift
    before_peak_index = peak_index - int(track_length / 10. / 2. / dx)
    after_peak_index = peak_index + int(track_length / 10. / 2. / dx)
    area_around_peak = np.trapz(extended_ramp[before_peak_index:after_peak_index], dx=dx)
    for i in range(before_peak_index + 1, after_peak_index):
        this_area = np.trapz(extended_ramp[before_peak_index:i], dx=dx)
        if this_area / area_around_peak >= 0.5:
            center_of_mass_index = i
            break
    center_of_mass_val = np.mean(extended_ramp[before_peak_index:after_peak_index])

    if extended_interp_x[center_of_mass_index] > induction_loc + 30.:
        center_of_mass_index -= len(interp_ramp)
    center_of_mass_x = extended_interp_x[center_of_mass_index]
    start_index = np.where(extended_ramp[:center_of_mass_index] <= 0.15 * center_of_mass_val)[0][-1]
    end_index = center_of_mass_index + np.where(extended_ramp[center_of_mass_index:] <= 0.15 * center_of_mass_val)[0][0]
    peak_shift = center_of_mass_x - induction_loc
    ramp_width = extended_interp_x[end_index] - extended_interp_x[start_index]
    before_width = induction_loc - extended_interp_x[start_index]
    after_width = extended_interp_x[end_index] - induction_loc
    ratio = before_width / after_width
    print '%s:' % title
    print '  amplitude: %.1f' % center_of_mass_val
    print '  peak_shift: %.1f' % peak_shift
    print '  ramp_width: %.1f' % ramp_width
    print '  rise:decay ratio: %.1f' % ratio
    if plot:
        plt.plot(default_interp_x, interp_ramp)


def process_plasticity_rule_continuous(output_filename, plot=False):
    """

    :param output_filename: str
    :param plot: bool
    """
    with h5py.File(data_dir + output_filename + '.hdf5', 'r') as f:
        for rule in f:
            for cell_id in f[rule]:
                track_length = f[rule][cell_id].attrs['track_length']
                if 'run_vel' in f[rule][cell_id].attrs:
                    run_vel = f[rule][cell_id].attrs['run_vel']
                else:
                    run_vel = 30.
                dt = f[rule][cell_id].attrs['dt']
                dx = dt * run_vel / 1000.
                induction_loc = np.mean(f[rule][cell_id].attrs['induction_loc'])
                ramp = f[rule][cell_id]['ramp'][:]
                model_ramp = f[rule][cell_id]['model_ramp'][:]
                for this_ramp, this_ramp_title in zip((ramp, model_ramp), ('exp', 'model')):
                    x = np.arange(track_length / len(this_ramp) / 2., track_length, track_length / len(this_ramp))
                    print_ramp_features(x, this_ramp, this_ramp_title+'_'+rule+cell_id, track_length, dx, induction_loc,
                                        plot)
                if plot:
                    plt.title(cell_id)
                    plt.show()
                    plt.close()


class optimize_history(object):
    def __init__(self):
        """

        """
        self.xlabels = []
        self.x_values = []
        self.error_values = []
        self.features = {}

    def report_best(self):
        """
        Report the input parameters and output values with the lowest error.
        :param feature: string
        :return:
        """
        lowest_Err = min(self.error_values)
        index = self.error_values.index(lowest_Err)
        best_x = self.x_values[index]
        formatted_x = '[' + ', '.join(['%.3E' % xi for xi in best_x]) + ']'
        print 'best x: %s' % formatted_x
        print 'lowest Err: %.3E' % lowest_Err
        return best_x

    def export_to_pkl(self, hist_filename):
        """
        Save the history to .pkl
        :param hist_filename: str
        """
        saved_history = {'xlabels': self.xlabels, 'x_values': self.x_values, 'error_values': self.error_values,
                         'features': self.features}
        write_to_pkl(data_dir+hist_filename+'.pkl', saved_history)

    def import_from_pkl(self, hist_filename):
        """
        Update a history object with data from a .pkl file
        :param hist_filename: str
        """
        previous_history = read_from_pkl(data_dir+hist_filename +'.pkl')
        self.xlabels = previous_history['xlabels']
        #self.xlabels = ['soma.g_pas', 'dend.g_pas slope']
        self.x_values = previous_history['x_values']
        self.error_values = previous_history['error_values']
        self.features = previous_history['features']

    def plot(self):
        """
        Plots each value in x_values against error
        """
        num_x_param = len(self.xlabels)
        num_plot_rows = math.floor(math.sqrt(num_x_param))
        num_plot_cols = math.ceil(num_x_param/num_plot_rows)

        #plot x-values against error
        plt.figure(1)
        for i, x_param in enumerate(self.xlabels):
            plt.subplot(num_plot_rows, num_plot_cols, i+1)
            x_param_vals = [x_val[i] for x_val in self.x_values]
            range_param_vals = max(x_param_vals) - min(x_param_vals)
            plt.scatter(x_param_vals, self.error_values)
            plt.xlim((min(x_param_vals)-0.1*range_param_vals, max(x_param_vals)+0.1*range_param_vals))
            plt.xlabel(x_param)
            plt.ylabel("Error values")
        plt.show()
        plt.close()

    def plot_features(self, feat_list=None, x_indices=None):
        if feat_list is None:
            feat_list = self.features.keys()
        if x_indices is None:
            x_indices = range(0, len(self.xlabels))
        num_x_param = len(x_indices)
        num_plot_rows = math.floor(math.sqrt(num_x_param))
        num_plot_cols = math.ceil(num_x_param/num_plot_rows)

        for i, feature in enumerate(feat_list):
            plt.figure(i+1)
            for index in x_indices:
                plt.subplot(num_plot_rows, num_plot_cols, i+1)
                x_param_vals = [x_val[index] for x_val in self.x_values]
                range_param_vals = max(x_param_vals) - min(x_param_vals)
                plt.scatter([x_param_vals], self.features[feature])
                plt.xlim((min(x_param_vals) - 0.1 * range_param_vals, max(x_param_vals) + 0.1 * range_param_vals))
                plt.xlabel(self.xlabels[index])
                plt.ylabel(feature)
                plt.legend(loc='upper right', scatterpoints=1, frameon=False, framealpha=0.5)
        plt.show()
        plt.close()


def sigmoid(p, x):
    x0, y0, c, k = p
    y = c / (1. + np.exp(-k * (x - x0))) + y0
    return y


class Pr(object):
    """
    This object contains internal variables to track the evolution in time of parameters governing synaptic release
    probability, used during optimization, and then exported to pr.mod for use during patterned input simulations.
    """
    def __init__(self, P0, f, tau_F, d, tau_D):
        self.P0 = P0
        self.f = f
        self.tau_F = tau_F
        self.d = d
        self.tau_D = tau_D
        self.P = P0
        self.tlast = None
        self.F = 1.
        self.D = 1.

    def stim(self, stim_time):
        """
        Evolve the dynamics until the current stim_time, report the current P, and update the internal parameters.
        :param stim_time: float
        :return: float
        """
        if self.tlast is not None:
            self.F = 1. + (self.F - 1.) * np.exp(-(stim_time - self.tlast) / self.tau_F)
            self.D = 1. - (1. - self.D) * np.exp(-(stim_time - self.tlast) / self.tau_D)
            self.P = min(1., self.P0 * self.F * self.D)
        self.tlast = stim_time
        self.F += self.f
        self.D *= self.d
        return self.P


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()


def list_find (f, lst):
    i = 0
    for x in lst:
        if f(x):
            return i
        else:
            i += 1
    return None


class StateMachine(object):
    """

    """

    def __init__(self, ti=0., dt=1., states=None, rates=None):
        """

        :param ti: float 
        :param dt: float
        :param states: dict
        :param rates: dict
        """
        self.dt = dt
        self.init_states = dict()
        self.states = dict()
        self.states_history = dict()
        self.rates = dict()
        if states is not None:
            self.update_states(states)
        if rates is not None:
            self.update_rates(rates)  # {'A': {'B': constant or iterable}}
        self.ti = ti
        self.t = self.ti
        self.t_history = np.array([self.t])
        self.i = 0

    def reset(self):
        """

        :param ti: 
        :return: 
        """
        self.t = self.ti
        self.i = 0
        self.t_history = np.array([self.t])
        self.states = dict(self.init_states)
        for s0 in self.states:
            self.states_history[s0] = np.array([self.states[s0]])

    def get_current_rates(self):
        """

        :return: dict
        """
        current = {}
        for s0 in self.rates:
            if s0 not in current:
                current[s0] = {}
            for s1 in self.rates[s0]:
                r = self.rates[s0][s1]
                if hasattr(r, '__iter__'):
                    if len(r) - 1 < self.i:
                        raise Exception('StateMachine: Insufficient array length for non-stationary rate: %s to %s ' %
                                        (s0, s1))
                    this_r = r[self.i]
                else:
                    this_r = r
                current[s0][s1] = this_r
        return current

    def update_transition(self, s0, s1, r):
        """

        :param s0: str
        :param s1: str
        :param r: float or array
        """
        if s0 not in self.states:
            raise Exception('StateMachine: Cannot update transition from invalid state: %s' % s0)
        if s1 not in self.states:
            raise Exception('StateMachine: Cannot update transition to invalid state: %s' % s1)
        if s0 not in self.rates:
            self.rates[s0] = {}
        self.rates[s0][s1] = r

    def update_rates(self, rates):
        """

        :param rates: dict  
        """
        for s0 in rates:
            for s1, r in rates[s0].iteritems():
                self.update_transition(s0, s1, r)

    def update_states(self, states):
        """

        :param states: dict
        """
        for s, v in states.iteritems():
            self.init_states[s] = v
            self.states[s] = v
            self.states_history[s] = np.array([v])

    def get_out_rate(self, state):
        """

        :param state: str
        :return: float 
        """
        if state not in self.states:
            raise Exception('StateMachine: Invalid state: %s' % state)
        if state not in self.rates:
            raise Exception('StateMachine: State: %s has no outgoing transitions' % state)
        out_rate = 0.
        for s1 in self.rates[state]:
            r = self.rates[state][s1]
            if hasattr(r, '__iter__'):
                if len(r) - 1 < self.i:
                    raise Exception('StateMachine: Insufficient array length for non-stationary rate: %s to %s ' %
                                    (state, s1))
                this_r = r[self.i]
            else:
                this_r = r
            out_rate += this_r
        return out_rate

    def step(self, n=1):
        """

        :param n: int 
        """
        for i in range(n):
            next_states = dict(self.states)
            for s0 in self.rates:
                total_out_prob = self.get_out_rate(s0) * self.dt
                if total_out_prob > 1.:
                    factor = 1. / total_out_prob
                else:
                    factor = 1.
                for s1 in self.rates[s0]:
                    r = self.rates[s0][s1]
                    if hasattr(r, '__iter__'):
                        if len(r) - 1 < self.i:
                            raise Exception('StateMachine: Insufficient array length for non-stationary rate: %s to '
                                            '%s ' % (s0, s1))
                        this_r = r[self.i]
                    else:
                        this_r = r
                    # print 'this_r: %.4E, factor: %.4E, %s: %.4E' % (this_r, factor, s0, self.states[s0])
                    this_delta = this_r * self.dt * factor * self.states[s0]
                    next_states[s0] -= this_delta
                    next_states[s1] += this_delta
            self.states = dict(next_states)
            for s0 in self.states:
                self.states_history[s0] = np.append(self.states_history[s0], self.states[s0])
            self.i += 1
            self.t += self.dt
            self.t_history = np.append(self.t_history, self.t)

    def run(self):
        """

        """
        self.reset()
        min_steps = None
        for s0 in self.rates:
            for s1 in self.rates[s0]:
                r = self.rates[s0][s1]
                if hasattr(r, '__iter__'):
                    if min_steps is None:
                        min_steps = len(r)
                    else:
                        min_steps = min(min_steps, len(r))
        if min_steps is None:
            raise Exception('StateMachine: Use step method to specify number of steps for stationary process.')
        self.step(min_steps)

    def plot(self, states=None):
        """

        :param states:
        """
        if states is None:
            states = self.states.keys()
        elif not hasattr(states, '__iter__'):
            states = [states]
        fig, axes = plt.subplots(1)
        for state in states:
            if state in self.states:
                axes.plot(self.t_history, self.states_history[state], label=state)
            else:
                print 'StateMachine: Not including invalid state: %s' % state
        axes.set_xlabel('Time (ms)')
        axes.set_ylabel('Occupancy')
        axes.legend(loc='best', frameon=False, framealpha=0.5)
        clean_axes(axes)
        plt.show()
        plt.close()


def flush_engine_buffer(result):
    """
    Once an async_result is ready, print the contents of its stdout buffer.
    :param result: :class:'ASyncResult
    """
    for stdout in result.stdout:
        if stdout:
            for line in stdout.splitlines():
                print line
    sys.stdout.flush()


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