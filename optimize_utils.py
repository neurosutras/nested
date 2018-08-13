"""
Library of functions and classes to support nested.optimize
"""
__author__ = 'Aaron D. Milstein and Grace Ng'
from nested.utils import *
import collections
from scipy._lib._util import check_random_state
from copy import deepcopy

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import BallTree, DistanceMetric
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import math


class Individual(object):
    """

    """
    def __init__(self, x):
        """

        :param x: array
        """
        self.x = np.array(x)
        self.features = None
        self.objectives = None
        self.energy = None
        self.rank = None
        self.distance = None
        self.fitness = None
        self.survivor = False


class PopulationStorage(object):
    """
    Class used to store populations of parameters and objectives during optimization.
    """
    def __init__(self, param_names=None, feature_names=None, objective_names=None, path_length=None, file_path=None):
        """

        :param param_names: list of str
        :param feature_names: list of str
        :param objective_names: list of str
        :param path_length: int
        :param file_path: str (path)
        """
        if file_path is not None:
            if os.path.isfile(file_path):
                self.load(file_path)
            else:
                raise IOError('PopulationStorage: invalid file path: %s' % file_path)
        else:
            if isinstance(param_names, collections.Iterable) and isinstance(feature_names, collections.Iterable) and \
                    isinstance(objective_names, collections.Iterable):
                self.param_names = param_names
                self.feature_names = feature_names
                self.objective_names = objective_names
            else:
                raise TypeError('PopulationStorage: names of params, features, and objectives must be specified as '
                                'lists')
            if type(path_length) == int:
                self.path_length = path_length
            else:
                raise TypeError('PopulationStorage: path_length must be specified as int')
            self.history = []  # a list of populations, each corresponding to one generation
            self.survivors = []  # a list of populations (some may be empty)
            self.failed = []  # a list of populations (some may be empty)
            self.min_objectives = None
            self.max_objectives = None
            # Enable tracking of param_gen-specific attributes through kwargs to 'append'
            self.attributes = {}

    def append(self, population, survivors=None, failed=None, **kwargs):
        """

        :param population: list of :class:'Individual'
        :param survivors: list of :class:'Individual'
        :param failed: list of :class:'Individual'
        :param kwargs: dict of additional param_gen-specific attributes
        """
        if survivors is None:
            survivors = []
        if failed is None:
            failed = []
        self.survivors.append(deepcopy(survivors))
        self.history.append(deepcopy(population))
        self.failed.append(deepcopy(failed))
        self.min_objectives, self.max_objectives = get_objectives_edges(self.history[-1], self.min_objectives,
                                                                        self.max_objectives)
        for key in kwargs:
            if key not in self.attributes:
                self.attributes[key] = []
        for key in self.attributes:
            if key in kwargs:
                self.attributes[key].append(kwargs[key])
            else:
                self.attributes[key].append(None)

    def get_best(self, n=1, iterations=None, offset=None, evaluate=None, modify=False):
        """
        If iterations is specified as an integer q, compute new rankings for the last q iterations, including the set
        of survivors produced penultimate to the qth iteration.
        If 'all' iterations is specified, collapse across all iterations, exclude copies of Individuals that survived
        across iterations, and compute new global rankings.
        Return the n best.
        If modify is True, allow changes to the rankings of Individuals stored in history, otherwise operate on and
        discard copies.
        :param n: int or 'all'
        :param iterations: str or int
        :param offset: int
        :param evaluate: callable
        :param modify: bool
        :return: list of :class:'Individual'
        """
        if iterations is None or not (iterations in ['all', 'last'] or type(iterations) == int):
            iterations = 'last'
            print 'PopulationStorage: Defaulting to get_best in last iteration.'
        elif type(iterations) == int and iterations * self.path_length > len(self.history):
            iterations = 'all'
            print 'PopulationStorage: Defaulting to get_best across all iterations.'
        if offset is None or not type(offset) == int or offset >= len(self.history):
            offset = len(self.history) - 1
        end = offset + 1
        extra_generations = end % self.path_length
        if iterations == 'all':
            start = 0
        elif iterations == 'last':
            start = end - self.path_length - extra_generations
        else:
            start = end - iterations * self.path_length - extra_generations
        if evaluate is None:
            evaluate = evaluate_population_annealing
        elif isinstance(evaluate, collections.Callable):
            pass
        elif type(evaluate) == str and evaluate in globals() and isinstance(globals()[evaluate], collections.Callable):
            evaluate_name = evaluate
            evaluate = globals()[evaluate_name]
        else:
            raise TypeError('PopulationStorage: evaluate must be callable.')
        if modify:
            group = [individual for population in self.history[start:end] for individual in population]
            if start > 0:
                group.extend([individual for individual in self.survivors[start-1]])
        else:
            group = [deepcopy(individual) for population in self.history[start:end] for individual in population]
            if start > 0:
                group.extend([deepcopy(individual) for individual in self.survivors[start-1]])
        evaluate(group, self.min_objectives, self.max_objectives)
        group = sort_by_rank(group)
        if n == 'all':
            return group
        else:
            return group[:n]

    def plot(self):
        """

        """
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import cm
        import matplotlib as mpl
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        mpl.rcParams['svg.fonttype'] = 'none'
        mpl.rcParams['text.usetex'] = False
        cmap = cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=len(self.history))
        colors = list(cmap(np.linspace(0, 1, len(self.history))))
        for this_attr in ['fitness', 'energy', 'distance', 'survivor']:
            fig, axes = plt.subplots(1)
            for j, population in enumerate(self.history):
                pop_ranks = []
                pop_vals = []
                survivor_ranks = []
                survivor_vals = []
                for indiv in population:
                    if indiv.survivor:
                        survivor_ranks.append(indiv.rank)
                        survivor_vals.append(getattr(indiv, this_attr))
                    else:
                        pop_ranks.append(indiv.rank)
                        pop_vals.append(getattr(indiv, this_attr))
                axes.scatter(pop_ranks, pop_vals, c='None', edgecolors=colors[j], alpha=0.05)
                axes.scatter(survivor_ranks, survivor_vals, c=colors[j], alpha=0.2)
                axes.set_xlabel('Ranked individuals per iteration')
            if this_attr == 'energy':
                axes.set_title('relative ' + this_attr)
            else:
                axes.set_title(this_attr)
            divider = make_axes_locatable(axes)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
            cbar.set_label('Generation')
            clean_axes(axes)
            fig.show()
        fig, axes = plt.subplots(1)
        pop_size = 0
        num_survivors = 0
        this_attr = 'objectives'
        for j, population in enumerate(self.history):
            pop_size = max(pop_size, len(population))
            num_survivors = max(num_survivors, len(self.survivors[j]))
            pop_ranks = []
            pop_vals = []
            survivor_ranks = []
            survivor_vals = []
            for indiv in population:
                if indiv.survivor:
                    survivor_ranks.append(indiv.rank)
                    survivor_vals.append(np.sum(getattr(indiv, this_attr)))
                else:
                    pop_ranks.append(indiv.rank)
                    pop_vals.append(np.sum(getattr(indiv, this_attr)))
            axes.scatter(pop_ranks, pop_vals, c='None', edgecolors=colors[j], alpha=0.05)
            axes.scatter(survivor_ranks, survivor_vals, c=colors[j], alpha=0.2)
        divider = make_axes_locatable(axes)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
        cbar.set_label('Generation')
        clean_axes(axes)
        axes.set_xlabel('Ranked individuals per iteration')
        axes.set_title('absolute energy')
        fig.show()
        for i, param_name in enumerate(self.param_names):
            this_attr = 'x'
            fig, axes = plt.subplots(1)
            for j, population in enumerate(self.history):
                pop_ranks = []
                pop_vals = []
                survivor_ranks = []
                survivor_vals = []
                for indiv in population:
                    if indiv.survivor:
                        survivor_ranks.append(indiv.rank)
                        survivor_vals.append(getattr(indiv, this_attr)[i])
                    else:
                        pop_ranks.append(indiv.rank)
                        pop_vals.append(getattr(indiv, this_attr)[i])
                axes.scatter(pop_ranks, pop_vals, c='None', edgecolors=colors[j], alpha=0.05)
                axes.scatter(survivor_ranks, survivor_vals, c=colors[j], alpha=0.2)
                if len(self.failed[j]) > 0:
                    failed_ranks = [self.path_length * pop_size + num_survivors] * len(self.failed[j])
                    failed_vals = [getattr(indiv, this_attr)[i] for indiv in self.failed[j]]
                    axes.scatter(failed_ranks, failed_vals, c='grey', alpha=0.2)
            axes.set_xlabel('Ranked individuals per iteration')
            axes.set_title(param_name)
            divider = make_axes_locatable(axes)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
            cbar.set_label('Generation')
            clean_axes(axes)
            fig.show()
        for i, objective_name in enumerate(self.objective_names):
            this_attr = 'objectives'
            fig, axes = plt.subplots(1)
            for j, population in enumerate(self.history):
                pop_ranks = []
                pop_vals = []
                survivor_ranks = []
                survivor_vals = []
                for indiv in population:
                    if indiv.survivor:
                        survivor_ranks.append(indiv.rank)
                        survivor_vals.append(getattr(indiv, this_attr)[i])
                    else:
                        pop_ranks.append(indiv.rank)
                        pop_vals.append(getattr(indiv, this_attr)[i])
                axes.scatter(pop_ranks, pop_vals, c='None', edgecolors=colors[j], alpha=0.05)
                axes.scatter(survivor_ranks, survivor_vals, c=colors[j], alpha=0.2)
            axes.set_title(this_attr+': '+objective_name)
            axes.set_xlabel('Ranked individuals per iteration')
            divider = make_axes_locatable(axes)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
            cbar.set_label('Generation')
            clean_axes(axes)
            fig.show()
        for i, feature_name in enumerate(self.feature_names):
            this_attr = 'features'
            fig, axes = plt.subplots(1)
            for j, population in enumerate(self.history):
                pop_ranks = []
                pop_vals = []
                survivor_ranks = []
                survivor_vals = []
                for indiv in population:
                    if indiv.survivor:
                        survivor_ranks.append(indiv.rank)
                        survivor_vals.append(getattr(indiv, this_attr)[i])
                    else:
                        pop_ranks.append(indiv.rank)
                        pop_vals.append(getattr(indiv, this_attr)[i])
                axes.scatter(pop_ranks, pop_vals, c='None', edgecolors=colors[j], alpha=0.05)
                axes.scatter(survivor_ranks, survivor_vals, c=colors[j], alpha=0.2)
            axes.set_title(this_attr+': '+feature_name)
            axes.set_xlabel('Ranked individuals per iteration')
            divider = make_axes_locatable(axes)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
            cbar.set_label('Generation')
            clean_axes(axes)
            fig.show()

    def nan2None(self, attr):
        """
        Convert from numpy nan to Python None.
        :param attr: any
        :return: any
        """
        if np.isnan(attr):
            return None
        else:
            return attr

    def None2nan(self, attr):
        """
        Convert from Python None to numpy nan.
        :param attr: any
        :return: any
        """
        if attr is None:
            return np.nan
        else:
            return attr

    def save(self, file_path, n=None):
        """
        Adds data from the most recent n generations to the hdf5 file.
        :param file_path: str
        :param n: str or int
        """
        io = 'w' if n == 'all' else 'a'
        with h5py.File(file_path, io) as f:
            if 'param_names' not in f.attrs.keys():
                f.attrs['param_names'] = self.param_names
            if 'feature_names' not in f.attrs.keys():
                f.attrs['feature_names'] = self.feature_names
            if 'objective_names' not in f.attrs.keys():
                f.attrs['objective_names'] = self.objective_names
            if 'path_length' not in f.attrs.keys():
                f.attrs['path_length'] = self.path_length
            if n is None:
                n = 1
            elif n == 'all':
                n = len(self.history)
            elif not isinstance(n, int):
                n = 1
                print 'PopulationStorage: defaulting to exporting last generation to file.'
            gen_index = len(self.history) - n
            j = n
            while n > 0:
                if str(gen_index) in f:
                    print 'PopulationStorage: generation %s already exported to file.'
                else:
                    f.create_group(str(gen_index))
                    for key in self.attributes:
                        f[str(gen_index)].attrs[key] = self.attributes[key][gen_index]
                    for group_name, population in zip(['population', 'survivors', 'failed'],
                                                      [self.history[gen_index], self.survivors[gen_index],
                                                       self.failed[gen_index]]):
                        f[str(gen_index)].create_group(group_name)
                        for i, individual in enumerate(population):
                            f[str(gen_index)][group_name].create_group(str(i))
                            if group_name is not 'failed':
                                f[str(gen_index)][group_name][str(i)].attrs['energy'] = self.None2nan(individual.energy)
                                f[str(gen_index)][group_name][str(i)].attrs['rank'] = self.None2nan(individual.rank)
                                f[str(gen_index)][group_name][str(i)].attrs['distance'] = \
                                    self.None2nan(individual.distance)
                                f[str(gen_index)][group_name][str(i)].attrs['fitness'] = \
                                    self.None2nan(individual.fitness)
                                f[str(gen_index)][group_name][str(i)].attrs['survivor'] = \
                                    self.None2nan(individual.survivor)
                                f[str(gen_index)][group_name][str(i)].create_dataset('features',
                                                                                     data=[self.None2nan(val) for val in
                                                                                           individual.features],
                                                                                     compression='gzip',
                                                                                     compression_opts=9)
                                f[str(gen_index)][group_name][str(i)].create_dataset('objectives',
                                                                                     data=[self.None2nan(val) for val in
                                                                                           individual.objectives],
                                                                                     compression='gzip',
                                                                                     compression_opts=9)
                            f[str(gen_index)][group_name][str(i)].create_dataset('x',
                                                                                 data=[self.None2nan(val) for val in
                                                                                       individual.x],
                                                                                 compression='gzip', compression_opts=9)
                n -= 1
                gen_index += 1
        print 'PopulationStorage: saved %i generations (up to generation %i) to file: %s' % (j, gen_index-1, file_path)

    def load(self, file_path):
        """

        :param file_path: str
        """
        if not os.path.isfile(file_path):
            raise IOError('PopulationStorage: invalid file path: %s' % file_path)
        self.history = []  # a list of populations, each corresponding to one generation
        self.survivors = []  # a list of populations (some may be empty)
        self.failed = []  # a list of populations (some may be empty)
        self.min_objectives = None
        self.max_objectives = None
        self.attributes = {}  # a dict containing lists of param_gen-specific attributes
        with h5py.File(file_path, 'r') as f:
            self.param_names = f.attrs['param_names']
            self.feature_names = f.attrs['feature_names']
            self.objective_names = f.attrs['objective_names']
            self.path_length = f.attrs['path_length']
            for gen_index in xrange(len(f)):
                for key, value in f[str(gen_index)].attrs.iteritems():
                    if key not in self.attributes:
                        self.attributes[key] = []
                    self.attributes[key].append(value)
                history, survivors, failed = [], [], []
                for group_name, population in zip(['population', 'survivors', 'failed'], [history, survivors,
                                                                                          failed]):
                    group = f[str(gen_index)][group_name]
                    for i in xrange(len(group)):
                        ind_data = group[str(i)]
                        individual = Individual(ind_data['x'][:])
                        if group_name is not 'failed':
                            individual.features = ind_data['features'][:]
                            individual.objectives = ind_data['objectives'][:]
                            individual.energy = self.nan2None(ind_data.attrs['energy'])
                            individual.rank = self.nan2None(ind_data.attrs['rank'])
                            individual.distance = self.nan2None(ind_data.attrs['distance'])
                            individual.fitness = self.nan2None(ind_data.attrs['fitness'])
                            individual.survivor = self.nan2None(ind_data.attrs['survivor'])
                        population.append(individual)
                self.min_objectives, self.max_objectives = get_objectives_edges(history, self.min_objectives,
                                                                                self.max_objectives)
                self.history.append(history)
                self.survivors.append(survivors)
                self.failed.append(failed)
        print 'PopulationStorage: loaded %i generations from file: %s' % (len(self.history), file_path)


class RelativeBoundedStep(object):
    """
    Step-taking method for use with PopulationAnnealing. Steps each parameter within specified absolute and/or relative
    bounds. Explores the range in log10 space when the range is >= 2 orders of magnitude (except if the range spans
    zero. If bounds are not provided for some parameters, the default is (0.1 * x0, 10. * x0).
    """
    def __init__(self, x0=None, param_names=None, bounds=None, rel_bounds=None, stepsize=0.5, wrap=False, random=None,
                 disp=False, **kwargs):
        """

        :param x0: array
        :param param_names: list
        :param bounds: list of tuple
        :param rel_bounds: list of lists
        :param stepsize: float in [0., 1.]
        :param wrap: bool  # whether or not to wrap around bounds
        :param random: int or :class:'np.random.RandomState'
        :param disp: bool
        """
        self.disp = disp
        self.wrap = wrap
        self.stepsize = stepsize
        if x0 is None and bounds is None:
            raise ValueError('RelativeBoundedStep: Either starting parameters or bounds are missing.')
        if random is None:
            self.random = np.random
        else:
            self.random = random
        if param_names is None and rel_bounds is not None:
            raise ValueError('RelativeBoundedStep: Parameter names must be specified to parse relative bounds.')
        self.param_names = param_names
        self.param_indexes = {param: i for i, param in enumerate(param_names)}
        if bounds is None:
            xmin = [None for xi in x0]
            xmax = [None for xi in x0]
        else:
            xmin = [bound[0] for bound in bounds]
            xmax = [bound[1] for bound in bounds]
        if x0 is None:
            x0 = [None for i in xrange(len(bounds))]
        for i in xrange(len(x0)):
            if x0[i] is None:
                if xmin[i] is None or xmax[i] is None:
                    raise ValueError('RelativeBoundedStep: Either starting parameters or bounds are missing.')
                else:
                    x0[i] = 0.5 * (xmin[i] + xmax[i])
            if xmin[i] is None:
                if x0[i] > 0.:
                    xmin[i] = 0.1 * x0[i]
                elif x0[i] == 0.:
                    xmin[i] = -1.
                else:
                    xmin[i] = 10. * x0[i]
            if xmax[i] is None:
                if x0[i] > 0.:
                    xmax[i] = 10. * x0[i]
                elif x0[i] == 0.:
                    xmax[i] = 1.
                else:
                    xmax[i] = 0.1 * x0[i]
        self.x0 = np.array(x0)

        self.xmin = np.array(xmin)
        self.xmax = np.array(xmax)
        self.x_range = np.subtract(self.xmax, self.xmin)
        self.logmod = lambda x, offset, factor: np.log10(x * factor + offset)
        self.logmod_inv = lambda logmod_x, offset, factor: ((10. ** logmod_x) - offset) / factor
        self.abs_order_mag = []
        for i in xrange(len(xmin)):
            xi_logmin, xi_logmax, offset, factor = self.logmod_bounds(xmin[i], xmax[i])
            self.abs_order_mag.append(xi_logmax - xi_logmin)
        self.rel_bounds = rel_bounds
        if not self.check_bounds(self.x0):
            raise ValueError('RelativeBoundedStep: Starting parameters are not within specified bounds.')

    def __call__(self, current_x, stepsize=None, wrap=None):
        """
        Take a step within bounds. If stepsize or wrap is specified for an individual call, it overrides the default.
        :param current_x: array
        :param stepsize: float in [0., 1.]
        :param wrap: bool
        :return: array
        """
        if stepsize is None:
            stepsize = self.stepsize
        if wrap is None:
            wrap = self.wrap
        x = np.array(current_x)
        for i in xrange(len(x)):
            if not self.xmax[i] >= self.xmin[i]:
                raise ValueError('RelativeBoundedStep: Misspecified bounds: xmin[%i] is not <= to xmax[%i].' % i)
            new_xi = self.generate_param(x[i], i, self.xmin[i], self.xmax[i], stepsize, wrap, self.disp)
            x[i] = new_xi
        if self.rel_bounds is not None:
            x = self.apply_rel_bounds(x, stepsize, self.rel_bounds, self.disp)
        return x

    def logmod_bounds(self, xi_min, xi_max):
        """

        :param xi_min: float
        :param xi_max: float
        :return: xi_logmin, xi_logmax, offset, factor
        """
        if xi_min < 0.:
            if xi_max < 0.:
                offset = 0.
                factor = -1.
            elif xi_max == 0.:
                factor = -1.
                this_order_mag = np.log10(xi_min * factor)
                if this_order_mag > 0.:
                    this_order_mag = math.ceil(this_order_mag)
                else:
                    this_order_mag = math.floor(this_order_mag)
                offset = 10. ** min(0., this_order_mag - 2)
            else:
                # If xi_min and xi_max are opposite signs, do not sample in log space; do linear sampling
                return 0., 0., None, None
            xi_logmin = self.logmod(xi_max, offset, factor)  # When the sign is flipped, the max and min will reverse
            xi_logmax = self.logmod(xi_min, offset, factor)
        elif xi_min == 0.:
            if xi_max == 0.:
                return 0., 0., None, None
            else:
                factor = 1.
                this_order_mag = np.log10(xi_max * factor)
                if this_order_mag > 0.:
                    this_order_mag = math.ceil(this_order_mag)
                else:
                    this_order_mag = math.floor(this_order_mag)
                offset = 10. ** min(0., this_order_mag - 2)
                xi_logmin = self.logmod(xi_min, offset, factor)
                xi_logmax = self.logmod(xi_max, offset, factor)
        else:
            offset = 0.
            factor = 1.
            xi_logmin = self.logmod(xi_min, offset, factor)
            xi_logmax = self.logmod(xi_max, offset, factor)
        return xi_logmin, xi_logmax, offset, factor

    def logmod_inv_bounds(self, xi_logmin, xi_logmax, offset, factor):
        """

        :param xi_logmin: float
        :param xi_logmax: float
        :param offset: float
        :param factor: float
        :return: xi_min, xi_max
        """
        if factor < 0.:
            xi_min = self.logmod_inv(xi_logmax, offset, factor)
            xi_max = self.logmod_inv(xi_logmin, offset, factor)
        else:
            xi_min = self.logmod_inv(xi_logmin, offset, factor)
            xi_max = self.logmod_inv(xi_logmax, offset, factor)
        return xi_min, xi_max

    def generate_param(self, xi, i, xi_min, xi_max, stepsize, wrap, disp=False):
        """

        :param xi: float
        :param i: int
        :param min: float
        :param max: float
        :param stepsize: float
        :param wrap: bool
        :return:
        """
        if xi_min == xi_max:
            return xi_min
        if self.abs_order_mag[i] <= 1.:
            new_xi = self.linear_step(xi, i, xi_min, xi_max, stepsize, wrap, disp)
        else:
            xi_logmin, xi_logmax, offset, factor = self.logmod_bounds(xi_min, xi_max)
            order_mag = min(xi_logmax - xi_logmin, self.abs_order_mag[i] * stepsize)
            if order_mag <= 1.:
                new_xi = self.linear_step(xi, i, xi_min, xi_max, stepsize, wrap, disp)
            else:
                new_xi = self.log10_step(xi, i, xi_logmin, xi_logmax, offset, factor, stepsize, wrap, disp)
        return new_xi

    def linear_step(self, xi, i, xi_min, xi_max, stepsize=None, wrap=None, disp=False):
        """
        Steps the specified parameter within the bounds according to the current stepsize.
        :param xi: float
        :param i: int
        :param stepsize: float in [0., 1.]
        :param wrap: bool
        :return: float
        """
        if stepsize is None:
            stepsize = self.stepsize
        if wrap is None:
            wrap = self.wrap
        step = stepsize * self.x_range[i] / 2.
        if disp:
            print 'Before: xi: %.4f, step: %.4f, xi_min: %.4f, xi_max: %.4f' % (xi, step, xi_min, xi_max)
        if wrap:
            step = min(step, xi_max - xi_min)
            delta = self.random.uniform(-step, step)
            new_xi = xi + delta
            if xi_min > new_xi:
                new_xi = max(xi_max - (xi_min - new_xi), xi_min)
            elif xi_max < new_xi:
                new_xi = min(xi_min + (new_xi - xi_max), xi_max)
        else:
            xi_min = max(xi_min, xi - step)
            xi_max = min(xi_max, xi + step)
            new_xi = self.random.uniform(xi_min, xi_max)
        if disp:
            print 'After: xi: %.4f, step: %.4f, xi_min: %.4f, xi_max: %.4f' % (new_xi, step, xi_min, xi_max)
        return new_xi

    def log10_step(self, xi, i, xi_logmin, xi_logmax, offset, factor, stepsize=None, wrap=None, disp=False):
        """
        Steps the specified parameter within the bounds according to the current stepsize.
        :param xi: float
        :param i: int
        :param xi_logmin: float
        :param xi_logmax: float
        :param offset: float.
        :param factor: float
        :param stepsize: float in [0., 1.]
        :param wrap: bool
        :return: float
        """
        if stepsize is None:
            stepsize = self.stepsize
        if wrap is None:
            wrap = self.wrap
        xi_log = self.logmod(xi, offset, factor)
        step = stepsize * self.abs_order_mag[i] / 2.
        if disp:
            print 'Before: log_xi: %.4f, step: %.4f, xi_logmin: %.4f, xi_logmax: %.4f' % (xi_log, step, xi_logmin,
                                                                                          xi_logmax)
        if wrap:
            step = min(step, xi_logmax - xi_logmin)
            delta = np.random.uniform(-step, step)
            step_xi_log = xi_log + delta
            if xi_logmin > step_xi_log:
                step_xi_log = max(xi_logmax - (xi_logmin - step_xi_log), xi_logmin)
            elif xi_logmax < step_xi_log:
                step_xi_log = min(xi_logmin + (step_xi_log - xi_logmax), xi_logmax)
            new_xi = self.logmod_inv(step_xi_log, offset, factor)
        else:
            step_xi_logmin = max(xi_logmin, xi_log - step)
            step_xi_logmax = min(xi_logmax, xi_log + step)
            new_xi_log = self.random.uniform(step_xi_logmin, step_xi_logmax)
            new_xi = self.logmod_inv(new_xi_log, offset, factor)
        if disp:
            print 'After: xi: %.4f, step: %.4f, xi_logmin: %.4f, xi_logmax: %.4f' % (new_xi, step, xi_logmin,
                                                                                      xi_logmax)
        return new_xi

    def apply_rel_bounds(self, x, stepsize, rel_bounds=None, disp=False):
        """

        :param x: array
        :param stepsize: float
        :param rel_bounds: list
        :param disp: bool
        """
        if disp:
            print 'orig x: %s' % str(x)
        new_x = np.array(x)
        new_min = deepcopy(self.xmin)
        new_max = deepcopy(self.xmax)
        if rel_bounds is not None:
            for i, rel_bound_rule in enumerate(rel_bounds):
                dep_param = rel_bound_rule[0]  #Dependent param: name of the parameter that may be modified
                dep_param_ind = self.param_indexes[dep_param]
                if dep_param_ind >= len(x):
                    raise Exception('Dependent parameter index is out of bounds for rule %d.' %i)
                factor = rel_bound_rule[2]
                ind_param = rel_bound_rule[3]  #Independent param: name of the parameter that sets the bounds
                ind_param_ind = self.param_indexes[ind_param]
                if ind_param_ind >= len(x):
                    raise Exception('Independent parameter index is out of bounds for rule %d.' %i)
                if rel_bound_rule[1] == "=":
                    new_xi = factor * new_x[ind_param_ind]
                    if (new_xi >= self.xmin[dep_param_ind]) and (new_xi < self.xmax[dep_param_ind]):
                        new_x[dep_param_ind] = new_xi
                    else:
                        raise Exception('Relative bounds rule %d contradicts fixed parameter bounds.' %i)
                    continue
                if disp:
                    print 'Before rel bound rule %i. xi: %.4f, min: %.4f, max: %.4f' % (i, new_x[dep_param_ind],
                                                                                        new_min[dep_param_ind],
                                                                                        new_max[dep_param_ind])

                if rel_bound_rule[1] == "<":
                    rel_max = factor * new_x[ind_param_ind]
                    new_max[dep_param_ind] = max(min(new_max[dep_param_ind], rel_max), new_min[dep_param_ind])
                elif rel_bound_rule[1] == "<=":
                    rel_max = factor * new_x[ind_param_ind]
                    new_max[dep_param_ind] = max(min(new_max[dep_param_ind], np.nextafter(rel_max, rel_max + 1)),
                                                 new_min[dep_param_ind])
                elif rel_bound_rule[1] == ">=":
                    rel_min = factor * new_x[ind_param_ind]
                    new_min[dep_param_ind] = min(max(new_min[dep_param_ind], rel_min), new_max[dep_param_ind])
                elif rel_bound_rule[1] == ">":
                    rel_min = factor * new_x[ind_param_ind]
                    new_min[dep_param_ind] = min(max(new_min[dep_param_ind], np.nextafter(rel_min, rel_min + 1)),
                                                 new_max[dep_param_ind])
                if not (new_min[dep_param_ind] <= new_x[dep_param_ind] < new_max[dep_param_ind]):
                    new_xi = max(new_x[dep_param_ind], new_min[dep_param_ind])
                    new_xi = min(new_xi, new_max[dep_param_ind])
                    if disp:
                        print 'After rel bound rule %i. xi: %.4f, min: %.4f, max: %.4f' % (i, new_xi,
                                                                                           new_min[dep_param_ind],
                                                                                           new_max[dep_param_ind])
                    new_x[dep_param_ind] = self.generate_param(new_xi, dep_param_ind, new_min[dep_param_ind],
                                                               new_max[dep_param_ind], stepsize, wrap=False, disp=disp)
        return new_x

    def check_bounds(self, x):
        """

        :param x: array
        :return: bool
        """
        # check absolute bounds first
        for i, xi in enumerate(x):
            if not (xi == self.xmin[i] and xi == self.xmax[i]):
                if xi < self.xmin[i]:
                    return False
                if xi > self.xmax[i]:
                    return False
        if self.rel_bounds is not None:
            for r, rule in enumerate(self.rel_bounds):
                # Dependent param. index: index of the parameter that may be modified
                dep_param_ind = self.param_indexes[rule[0]]
                if dep_param_ind >= len(x):
                    raise Exception('Dependent parameter index is out of bounds for rule %d.' % r)
                factor = rule[2]
                # Independent param. index: index of the parameter that sets the bounds
                ind_param_ind = self.param_indexes[rule[3]]
                if ind_param_ind >= len(x):
                    raise Exception('Independent parameter index is out of bounds for rule %d.' % r)
                if rule[1] == "=":
                    operator = lambda x, y: x == y
                elif rule[1] == "<":
                    operator = lambda x, y: x < y
                elif rule[1] == "<=":
                    operator = lambda x, y: x <= y
                elif rule[1] == ">=":
                    operator = lambda x, y: x >= y
                elif rule[1] == ">":
                    operator = lambda x, y: x > y
                if not operator(x[dep_param_ind], factor * x[ind_param_ind]):
                    print 'Parameter %d: value %.3f did not meet relative bound in rule %d.' % \
                          (dep_param_ind, x[dep_param_ind], r)
                    return False
        return True


class PopulationAnnealing(object):
    """
    This class is inspired by scipy.optimize.basinhopping. It provides a generator interface to produce a list of
    parameter arrays intended for parallel evaluation. Features multi-objective metrics for selection and adaptive
    reduction of step_size (or temperature) every iteration. During each iteration, each individual in the population
    takes path_length number of independent steps within the specified bounds, then individuals are selected to seed the
    population for the next iteration.
    """
    def __init__(self, param_names=None, feature_names=None, objective_names=None, pop_size=None, x0=None, bounds=None,
                 rel_bounds=None, wrap_bounds=False, take_step=None, evaluate=None, select=None, seed=None,
                 max_iter=50, path_length=3, initial_step_size=0.5, adaptive_step_factor=0.9, survival_rate=0.2,
                 max_fitness=5, disp=False, hot_start=False, storage_file_path=None,  **kwargs):
        """
        :param param_names: list of str
        :param feature_names: list of str
        :param objective_names: list of str
        :param pop_size: int
        :param x0: array
        :param bounds: list of tuple of float
        :param rel_bounds: list of list
        :param wrap_bounds: bool
        :param take_step: callable
        :param evaluate: callable
        :param select: callable
        :param seed: int or :class:'np.random.RandomState'
        :param max_iter: int
        :param path_length: int
        :param initial_step_size: float in [0., 1.]
        :param adaptive_step_factor: float in [0., 1.]
        :param survival_rate: float in [0., 1.]
        :param disp: bool
        :param hot_start: bool
        :param storage_file_path: str (path)
        :param kwargs: dict of additional options, catches generator-specific options that do not apply
        """
        if x0 is None:
            self.x0 = None
        else:
            self.x0 = np.array(x0)
        if evaluate is None:
            self.evaluate = evaluate_population_annealing
        elif isinstance(evaluate, collections.Callable):
            self.evaluate = evaluate
        elif type(evaluate) == str and evaluate in globals() and isinstance(globals()[evaluate], collections.Callable):
            self.evaluate = globals()[evaluate]
        else:
            raise TypeError("PopulationAnnealing: evaluate must be callable.")
        if select is None:
            # self.select = select_survivors_by_rank_and_fitness  # select_survivors_by_rank
            self.select = select_survivors_population_annealing
        elif isinstance(select, collections.Callable):
            self.select = select
        elif type(select) == str and select in globals() and isinstance(globals()[select], collections.Callable):
            self.select = globals()[select]
        else:
            raise TypeError("PopulationAnnealing: select must be callable.")
        self.random = check_random_state(seed)
        self.xmin = np.array([bound[0] for bound in bounds])
        self.xmax = np.array([bound[1] for bound in bounds])
        self.storage_file_path = storage_file_path
        if hot_start:
            if self.storage_file_path is None or not os.path.isfile(self.storage_file_path):
                raise IOError('PopulationAnnealing: invalid file path. Cannot hot start from stored history: %s' %
                              hot_start)
            else:
                self.storage = PopulationStorage(file_path=self.storage_file_path)
                param_names = self.storage.param_names
                self.path_length = self.storage.path_length
                if 'step_size' in self.storage.attributes:
                    current_step_size = self.storage.attributes['step_size'][-1]
                else:
                    current_step_size = None
                if current_step_size is not None:
                    initial_step_size = current_step_size
                self.num_gen = len(self.storage.history)
                self.population = self.storage.history[-1]
                self.survivors = self.storage.survivors[-1]
                self.failed = self.storage.failed[-1]
                self.objectives_stored = True
        else:
            self.storage = PopulationStorage(param_names=param_names, feature_names=feature_names,
                                             objective_names=objective_names, path_length=path_length)
            self.path_length = path_length
            self.num_gen = 0
            self.population = []
            self.survivors = []
            self.failed = []
            self.objectives_stored = False
        self.pop_size = pop_size
        if take_step is None:
            self.take_step = RelativeBoundedStep(self.x0, param_names=param_names, bounds=bounds, rel_bounds=rel_bounds,
                                                 stepsize=initial_step_size, wrap=wrap_bounds, random=self.random)
        elif isinstance(take_step, collections.Callable):
                self.take_step = take_step(self.x0, param_names=param_names, bounds=bounds,
                                                      rel_bounds=rel_bounds, stepsize=initial_step_size,
                                                      wrap=wrap_bounds, random=self.random)
        elif type(take_step) == str and take_step in globals() and \
                isinstance(globals()[take_step], collections.Callable):
                self.take_step = globals()[take_step](self.x0, param_names=param_names, bounds=bounds,
                                                      rel_bounds=rel_bounds, stepsize=initial_step_size,
                                                      wrap=wrap_bounds, random=self.random)
        else:
            raise TypeError('PopulationAnnealing: provided take_step: %s is not callable.' % take_step)
        self.x0 = np.array(self.take_step.x0)
        self.xmin = np.array(self.take_step.xmin)
        self.xmax = np.array(self.take_step.xmax)
        self.max_gens = self.path_length * max_iter
        self.adaptive_step_factor = adaptive_step_factor
        self.num_survivors = max(1, int(self.pop_size * survival_rate))
        self.max_fitness = max_fitness
        self.disp = disp
        self.local_time = time.time()

    def __call__(self):
        """
        A generator that yields a list of parameter arrays with size pop_size.
        :yields: list of :class:'Individual'
        """
        self.start_time = time.time()
        self.local_time = self.start_time
        while self.num_gen < self.max_gens:
            if self.num_gen == 0:
                self.init_population()
            elif not self.objectives_stored:
                raise Exception('PopulationAnnealing: objectives from previous Gen %i were not stored or evaluated' %
                                (self.num_gen-1))
            elif self.num_gen % self.path_length == 0:
                self.step_survivors()
            else:
                self.step_population()
            self.objectives_stored = False
            if self.disp:
                print 'PopulationAnnealing: Gen %i, yielding parameters for population size %i' % \
                      (self.num_gen, len(self.population))
            self.local_time = time.time()
            self.num_gen += 1
            sys.stdout.flush()
            yield [individual.x for individual in self.population]
        if not self.objectives_stored:
            raise Exception('PopulationAnnealing: objectives from final Gen %i were not stored or evaluated' %
                            (self.num_gen - 1))
        if self.disp:
            print 'PopulationAnnealing: %i generations took %.2f s' % (self.max_gens, time.time()-self.start_time)
        sys.stdout.flush()

    def update_population(self, features, objectives):
        """
        Expects a list of objective arrays to be in the same order as the list of parameter arrays yielded from the
        current generation.
        :param features: list of dict
        :param objectives: list of dict
        """
        filtered_population = []
        num_failed = 0
        for i, objective_dict in enumerate(objectives):
            feature_dict = features[i]
            if not isinstance(objective_dict, dict):
                raise TypeError('PopulationAnnealing.update_population: objectives must be a list of dict')
            if not isinstance(feature_dict, dict):
                raise TypeError('PopulationAnnealing.update_population: features must be a list of dict')
            if not (all(key in objective_dict for key in self.storage.objective_names) and
                    all(key in feature_dict for key in self.storage.feature_names)):
                self.failed.append(self.population[i])
                num_failed += 1
            else:
                this_objectives = np.array([objective_dict[key] for key in self.storage.objective_names])
                self.population[i].objectives = this_objectives
                this_features = np.array([feature_dict[key] for key in self.storage.feature_names])
                self.population[i].features = this_features
                filtered_population.append(self.population[i])
        if self.disp:
            print 'PopulationAnnealing: Gen %i, computing features for population size %i took %.2f s; %i individuals' \
                  ' failed' % (self.num_gen - 1, len(self.population), time.time() - self.local_time, num_failed)
        self.local_time = time.time()
        self.population = filtered_population
        self.storage.append(self.population, survivors=self.survivors, failed=self.failed,
                            step_size=self.take_step.stepsize)
        self.objectives_stored = True
        if self.num_gen % self.path_length == 0:
            self.select_survivors()
            self.storage.survivors[-1] = self.survivors
            if self.storage_file_path is not None:
                self.storage.save(self.storage_file_path, n=self.path_length)
        else:
            self.survivors = []
        sys.stdout.flush()

    def select_survivors(self):
        """

        """
        candidate_survivors = [individual for individual in
                               self.storage.get_best(n='all', iterations=1, evaluate=self.evaluate, modify=True) if
                               self.take_step.check_bounds(individual.x)]
        survivors, specialists = self.select(candidate_survivors, self.num_survivors)
        self.survivors = survivors + specialists
        for individual in self.survivors:
            individual.survivor = True
        if self.disp:
            print 'PopulationAnnealing: Gen %i, evaluating iteration took %.2f s' % (self.num_gen - 1,
                                                                                     time.time() - self.local_time)
        self.local_time = time.time()

    def init_population(self):
        """

        """
        pop_size = self.pop_size
        if self.x0 is not None:
            self.population = []
            self.population.append(Individual(self.x0))
            pop_size -= 1
            self.population.extend([Individual(self.take_step(self.x0, stepsize=1., wrap=True))
                                    for i in xrange(pop_size)])
        else:
            self.population = [Individual(x) for x in self.random.uniform(self.xmin, self.xmax, pop_size)]

    def step_survivors(self):
        """
        Consider the highest ranked Individuals of the previous iteration be survivors. Seed the next generation with
        steps taken from the set of survivors.
        """
        new_step_size = self.take_step.stepsize * self.adaptive_step_factor
        if self.disp:
            print 'PopulationAnnealing: Gen %i, previous step_size: %.3f, new step_size: %.3f' % \
                  (self.num_gen, self.take_step.stepsize, new_step_size)
        self.take_step.stepsize = new_step_size
        new_population = []
        if not self.survivors:
            self.init_population()
        else:
            num_survivors = min(self.num_survivors, len(self.survivors))
            for i in xrange(self.pop_size):
                individual = Individual(self.take_step(self.survivors[i % num_survivors].x))
                new_population.append(individual)
            self.population = new_population

    def step_population(self):
        """

        """
        this_pop_size = len(self.population)
        if this_pop_size == 0:
            self.init_population()
        else:
            new_population = []
            for i in xrange(self.pop_size):
                individual = Individual(self.take_step(self.population[i % this_pop_size].x))
                new_population.append(individual)
            self.population = new_population


class HallOfFame(object):
    """
    Convenience object to access parameters, features, and objectives for 'best' and 'specialist' Individuals following
    optimization.
    """
    def __init__(self, storage):
        """

        :param storage: :class:'PopulationStorage'
        """
        self.param_names = storage.param_names
        self.objective_names = storage.objective_names
        self.feature_names = storage.feature_names
        population = storage.get_best('all', 'last')
        survivors, specialists = select_survivors_population_annealing(population, 1)
        self.x_dict = dict()
        self.x_array = dict()
        self.features = dict()
        self.objectives = dict()
        self.append(survivors[0], 'best')
        for i, name in enumerate(self.objective_names):
            self.append(specialists[i], name)

    def append(self, individual, name):
        """

        :param individual: :class:'Individual'
        :param name: str
        """
        self.x_array[name] = individual.x
        self.x_dict[name] = param_array_to_dict(individual.x, self.param_names)
        self.features[name] = param_array_to_dict(individual.features, self.feature_names)
        self.objectives[name] = param_array_to_dict(individual.objectives, self.objective_names)

    def report(self, name):
        """

        :param name: str
        """
        if name not in self.x_array:
            raise KeyError('HallOfFame: no data associated with the name: %s' % name)
        print '%s:' % name
        print 'params:'
        pprint.pprint(self.x_dict[name])
        print 'features:'
        pprint.pprint(self.features[name])
        print 'objectives:'
        pprint.pprint(self.objectives[name])


def get_relative_energy(energy, min_energy, max_energy):
    """
    If the range of absolute energy values is within 2 orders of magnitude, translate and normalize linearly. Otherwise,
    translate and normalize based on the distance between values in log space.
    :param energy: array
    :param min_energy: float
    :param max_energy: float
    :return: array
    """
    logmod = lambda x, offset: np.log10(x + offset)
    if min_energy == 0.:
        this_order_mag = np.log10(max_energy)
        if this_order_mag > 0.:
            this_order_mag = math.ceil(this_order_mag)
        else:
            this_order_mag = math.floor(this_order_mag)
        offset = 10. ** min(0., this_order_mag - 2)
        logmin = logmod(min_energy, offset)
        logmax = logmod(max_energy, offset)
    else:
        offset = 0.
        logmin = logmod(min_energy, offset)
        logmax = logmod(max_energy, offset)
    logmod_range = logmax - logmin
    if logmod_range < 2.:
        energy_vals = np.subtract(energy, min_energy)
        energy_vals = np.divide(energy_vals, max_energy - min_energy)
    else:
        energy_vals = [logmod(energy_val, offset) for energy_val in energy]
        energy_vals = np.subtract(energy_vals, logmin)
        energy_vals = np.divide(energy_vals, logmod_range)
    return energy_vals
    

def get_objectives_edges(population, min_objectives=None, max_objectives=None):
    """

    :param population: list of :class:'Individual'
    :param min_objectives: array
    :param max_objectives: array
    :return: array
    """
    pop_size = len(population)
    num_objectives = [len(individual.objectives) for individual in population if individual.objectives is not None]
    if len(num_objectives) < pop_size:
        raise Exception('get_objectives_edges: objectives have not been stored for all Individuals in population')
    if min_objectives is None:
        this_min_objectives = np.array(population[0].objectives)
    else:
        this_min_objectives = np.array(min_objectives)
    if max_objectives is None:
        this_max_objectives = np.array(population[0].objectives)
    else:
        this_max_objectives = np.array(max_objectives)
    for individual in population:
        this_min_objectives = np.minimum(this_min_objectives, individual.objectives)
        this_max_objectives = np.maximum(this_max_objectives, individual.objectives)
    return this_min_objectives, this_max_objectives


def assign_crowding_distance(population):
    """
    Modifies in place the distance attribute of each Individual in the population.
    :param population: list of :class:'Individual'
    """
    pop_size = len(population)
    num_objectives = [len(individual.objectives) for individual in population if individual.objectives is not None]
    if len(num_objectives) < pop_size:
        raise Exception('assign_crowding_distance: objectives have not been stored for all Individuals in population')
    num_objectives = max(num_objectives)
    for individual in population:
        individual.distance = 0
    for m in xrange(num_objectives):
        indexes = range(pop_size)
        objective_vals = [individual.objectives[m] for individual in population]
        indexes.sort(key=objective_vals.__getitem__)
        new_population = map(population.__getitem__, indexes)

        # keep the borders
        new_population[0].distance += 1.e15
        new_population[-1].distance += 1.e15

        objective_min = new_population[0].objectives[m]
        objective_max = new_population[-1].objectives[m]

        if objective_min != objective_max:
            for i in xrange(1, pop_size - 1):
                new_population[i].distance += (new_population[i + 1].objectives[m] -
                                               new_population[i - 1].objectives[m]) / (objective_max - objective_min)


def sort_by_crowding_distance(population):
    """
    Sorts the population by the value of the distance attribute of each Individual in the population. Returns the sorted
    population.
    :param population: list of :class:'Individual'
    :return: list of :class:'Individual'
    """
    pop_size = len(population)
    distance_vals = [individual.distance for individual in population if individual.distance is not None]
    if len(distance_vals) < pop_size:
        raise Exception('sort_by_crowding_distance: crowding distance has not been stored for all Individuals in '
                        'population')
    indexes = range(pop_size)
    distances = [individual.distance for individual in population]
    indexes.sort(key=distances.__getitem__)
    indexes.reverse()
    population = map(population.__getitem__, indexes)
    return population


def assign_absolute_energy(population):
    """
    Modifies in place the energy attribute of each Individual in the population. Energy is assigned as the sum across
    all non-normalized objectives.
    :param population: list of :class:'Individual'
    """
    indexes = range(len(population))
    num_objectives = [len(individual.objectives) for individual in population if individual.objectives is not None]
    if len(num_objectives) < len(indexes):
        raise Exception('assign_absolute_energy: objectives have not been stored for all Individuals in population')
    for individual in population:
        individual.energy = np.sum(individual.objectives)


def sort_by_energy(population):
    """
    Sorts the population by the value of the energy attribute of each Individual in the population. Returns the sorted
    population.
    :param population: list of :class:'Individual'
    :return: list of :class:'Individual'
    """
    pop_size = len(population)
    energy_vals = [individual.energy for individual in population if individual.energy is not None]
    if len(energy_vals) < pop_size:
        raise Exception('sort_by_energy: energy has not been stored for all Individuals in population')
    indexes = range(pop_size)
    indexes.sort(key=energy_vals.__getitem__)
    population = map(population.__getitem__, indexes)
    return population


def assign_relative_energy(population, min_objectives=None, max_objectives=None):
    """
    Modifies in place the energy attribute of each Individual in the population. Each objective is normalized within
    the provided population. Energy is assigned as the sum across all normalized objectives.
    :param population: list of :class:'Individual'
    """
    pop_size = len(population)
    num_objectives = [len(individual.objectives) for individual in population if individual.objectives is not None]
    if len(num_objectives) < pop_size:
        raise Exception('assign_relative_energy: objectives have not been stored for all Individuals in population')
    num_objectives = max(num_objectives)
    for individual in population:
        individual.energy = 0
    if min_objectives is None or max_objectives is None:
        this_min_objectives, this_max_objectives = get_objectives_edges(population, min_objectives, max_objectives)
    else:
        this_min_objectives, this_max_objectives = np.array(min_objectives), np.array(max_objectives)
    for m in xrange(num_objectives):
        objective_vals = [individual.objectives[m] for individual in population]
        if this_min_objectives[m] != this_max_objectives[m]:
            energy_vals = get_relative_energy(objective_vals, this_min_objectives[m], this_max_objectives[m])
            for energy, individual in zip(energy_vals, population):
                individual.energy += energy


def assign_relative_energy_by_fitness(population):
    """
    Modifies in place the energy attribute of each Individual in the population. Each objective is normalized within
    each group of Individuals with equivalent fitness. Energy is assigned as the sum across all normalized objectives.
    :param population: list of :class:'Individual'
    """
    pop_size = len(population)
    fitness_vals = [individual.fitness for individual in population if individual.fitness is not None]
    if len(fitness_vals) < pop_size:
        raise Exception('assign_relative_energy_by_fitness: fitness has not been stored for all Individuals in '
                        'population')
    max_fitness = max(fitness_vals)
    for fitness in xrange(max_fitness + 1):
        new_front = [individual for individual in population if individual.fitness == fitness]
        assign_relative_energy(new_front)


def assign_rank_by_fitness_and_energy(population):
    """
    Deprecated.
    Modifies in place the rank attribute of each Individual in the population. Within each group of Individuals with
    equivalent fitness, sorts by the value of the energy attribute of each Individual.
    :param population: list of :class:'Individual'
    """
    pop_size = len(population)
    fitness_vals = [individual.fitness for individual in population if individual.fitness is not None]
    if len(fitness_vals) < pop_size:
        raise Exception('assign_rank_by_fitness_and_energy: fitness has not been stored for all Individuals in '
                        'population')
    max_fitness = max(fitness_vals)
    new_population = []
    for fitness in xrange(max_fitness + 1):
        new_front = [individual for individual in population if individual.fitness == fitness]
        new_sorted_front = sort_by_energy(new_front)
        new_population.extend(new_sorted_front)
    # now that population is sorted, assign rank to Individuals
    for rank, individual in enumerate(new_population):
        individual.rank = rank


def assign_rank_by_energy(population):
    """
    Modifies in place the rank attribute of each Individual in the population. Sorts by the value of the energy
    attribute of each Individual in the population.
    :param population: list of :class:'Individual'
    """
    new_population = sort_by_energy(population)
    # now that population is sorted, assign rank to Individuals
    for rank, individual in enumerate(new_population):
        individual.rank = rank


def sort_by_rank(population):
    """
    Sorts by the value of the rank attribute of each Individual in the population. Returns the sorted population.
    :param population: list of :class:'Individual'
    :return: list of :class:'Individual'
    """
    pop_size = len(population)
    rank_vals = [individual.rank for individual in population if individual.rank is not None]
    if len(rank_vals) < pop_size:
        raise Exception('sort_by_rank: rank has not been stored for all Individuals in population')
    indexes = range(pop_size)
    indexes.sort(key=rank_vals.__getitem__)
    new_population = map(population.__getitem__, indexes)
    return new_population


def assign_rank_by_fitness_and_crowding_distance(population):
    """
    TODO: Make 'assign_crowding_distance_by_fitness' first.
    Modifies in place the distance and rank attributes of each Individual in the population. This is appropriate for
    early generations of evolutionary optimization, and helps to preserve diversity of solutions. However, once all
    members of the population have converged to a single fitness value, naive ranking by crowding distance can favor
    unique solutions over lower energy solutions. In this case, rank is assigned by total energy.
    :param population: list of :class:'Individual'
    """
    pop_size = len(population)
    fitness_vals = [individual.fitness for individual in population if individual.fitness is not None]
    if len(fitness_vals) < pop_size:
        raise Exception('assign_rank_by_fitness_and_crowding_distance: fitness has not been stored for all Individuals '
                        'in population')
    max_fitness = max(fitness_vals)
    if max_fitness > 0:
        new_population = []
        for fitness in xrange(max_fitness + 1):
            new_front = [individual for individual in population if individual.fitness == fitness]
            assign_crowding_distance(new_front)
            new_sorted_front = sort_by_crowding_distance(new_front)
            new_population.extend(new_sorted_front)
    else:
        new_population = sort_by_energy(population)
    # now that population is sorted, assign rank to Individuals
    for rank, individual in enumerate(new_population):
        individual.rank = rank


def assign_fitness_by_dominance(population, disp=False):
    """
    Modifies in place the fitness attribute of each Individual in the population.
    :param population: list of :class:'Individual'
    :param disp: bool
    """
    def dominates(p, q):
        """
        Individual p dominates Individual q if each of its objective values is equal or better, and at least one of
        its objective values is better.
        :param p: :class:'Individual'
        :param q: :class:'Individual'
        :return: bool
        """
        diff12 = np.subtract(p.objectives, q.objectives)
        return ((diff12 <= 0.).all()) and ((diff12 < 0.).any())

    pop_size = len(population)
    num_objectives = [len(individual.objectives) for individual in population if individual.objectives is not None]
    if len(num_objectives) < pop_size:
        raise Exception('assign_fitness_by_dominance: objectives have not been stored for all Individuals in '
                        'population')
    num_objectives = max(num_objectives)
    if num_objectives > 1:
        F = {0: []}  # first front of dominant Individuals
        S = dict()
        n = dict()

        for p in xrange(len(population)):
            S[p] = []  # list of Individuals that p dominates
            n[p] = 0  # number of Individuals that dominate p

            for q in xrange(len(population)):
                if dominates(population[p], population[q]):
                    S[p].append(q)
                elif dominates(population[q], population[p]):
                    n[p] += 1

            if n[p] == 0:
                population[p].fitness = 0  # fitness 0 indicates first dominant front
                F[0].append(p)

        # excluding the Individuals that dominated the previous front, find the next front
        i = 0
        while len(F[i]) > 0:
            F[i+1] = []  # next front
            # take the elements from the previous front
            for p in F[i]:
                # take the elements that p dominates
                for q in S[p]:
                    # decrease domination value of all Individuals that p dominates
                    n[q] -= 1
                    if n[q] == 0:
                        population[q].fitness = i + 1  # assign fitness of current front
                        F[i+1].append(q)
            i += 1
    else:
        for individual in population:
            individual.fitness = 0
    if disp:
        print F


def evaluate_population_annealing(population, min_objectives=None, max_objectives=None, disp=False):
    """
    Modifies in place the fitness, energy and rank attributes of each Individual in the population.
    :param population: list of :class:'Individual'
    :param disp: bool
    """
    if len(population) > 0:
        assign_fitness_by_dominance(population)
        assign_relative_energy(population, min_objectives, max_objectives)
        assign_rank_by_fitness_and_energy(population)
    else:
        raise ValueError('evaluate_population_annealing: cannot evaluate empty population.')


def evaluate_random(population, disp=False):
    """
    Modifies in place the rank attribute of each Individual in the population.
    :param population: list of :class:'Individual'
    :param disp: bool
    """
    rank_vals = range(len(population))
    np.random.shuffle(rank_vals)
    for i, individual in enumerate(population):
        rank = rank_vals[i]
        individual.rank = rank
        if disp:
            print 'Individual %i: rank %i, x: %s' % (i, rank, individual.x)


def select_survivors_by_rank(population, num_survivors, disp=False, **kwargs):
    """
    Sorts the population by the rank attribute of each Individual in the population. Returns the requested number of
    top ranked Individuals.
    :param population: list of :class:'Individual'
    :param num_survivors: int
    :param disp: bool
    :return: list of :class:'Individual'
    """
    new_population = sort_by_rank(population)
    return new_population[:num_survivors]


def select_survivors_by_rank_and_fitness(population, num_survivors, max_fitness=None, disp=False, **kwargs):
    """
    Sorts the population by the rank attribute of each Individual in the population. Selects top ranked Individuals from
    each fitness group proportional to the size of each fitness group. Returns the requested number of Individuals.
    :param population: list of :class:'Individual'
    :param num_survivors: int
    :param max_fitness: int: select survivors with fitness values <= max_fitness
    :param disp: bool
    :return: list of :class:'Individual'
    """
    fitness_vals = [individual.fitness for individual in population if individual.fitness is not None]
    if len(fitness_vals) < len(population):
        raise Exception('select_survivors_by_rank_and_fitness: fitness has not been stored for all Individuals '
                        'in population')
    if max_fitness is None:
        max_fitness = max(fitness_vals)
    else:
        max_fitness = min(max(fitness_vals), max_fitness)
    pop_size = len(np.where(np.array(fitness_vals) <= max_fitness)[0])
    survivors = []
    for fitness in xrange(max_fitness + 1):
        new_front = [individual for individual in population if individual.fitness == fitness]
        sorted_front = sort_by_rank(new_front)
        this_num_survivors = max(1, int(math.ceil(float(len(sorted_front)) / float(pop_size) * num_survivors)))
        survivors.extend(sorted_front[:this_num_survivors])
        if len(survivors) >= num_survivors:
            return survivors[:num_survivors]
    return survivors


def select_survivors_population_annealing(population, num_survivors, get_specialists=True, disp=False, **kwargs):
    """
    Sorts the population by the rank attribute of each Individual in the population. Selects and returns the requested
    number of top ranked Individuals as well as a set of 'specialists' - individuals with the lowest objective error for
    one objective.
    :param population: list of :class:'Individual'
    :param num_survivors: int
    :param get_specialists: bool
    :param disp: bool
    :return: list of :class:'Individual'
    """
    new_population = sort_by_rank(population)
    survivors = new_population[:num_survivors]
    if get_specialists:
        pop_size = len(population)
        num_objectives = [len(individual.objectives) for individual in population if individual.objectives is not None]
        if len(num_objectives) < pop_size:
            raise Exception('assign_fitness_by_dominance: objectives have not been stored for all Individuals in '
                            'population')
        num_objectives = max(num_objectives)

        specialists = []
        for m in xrange(num_objectives):
            population = sorted(population, key=lambda individual: individual.objectives[m])
            group = []
            reference_objective_val = population[0].objectives[m]
            for individual in population:
                if individual.objectives[m] == reference_objective_val:
                    group.append(individual)
            if len(group) > 1:
                group = sorted(group, key=lambda individual: individual.energy)
            specialists.append(group[0])
        return survivors, specialists
    else:
        return survivors


def config_interactive(context, source_file_name, config_file_path=None, output_dir=None, temp_output_path=None,
                       export=False, export_file_path=None, label=None, disp=True, **kwargs):
    """
    nested.optimize is meant to be executed as a module, and refers to a config_file to import required submodules and
    create a workflow for optimization. During development of submodules, it is useful to be able to execute a submodule
    as a standalone script (as '__main__'). config_interactive allows a single process to properly parse the
    config_file and initialize a Context for testing purposes.
    :param context: :class:'Context'
    :param source_file_name: str (filename of calling module)
    :param config_file_path: str (.yaml file path)
    :param output_dir: str (dir path)
    :param temp_output_path: str (.hdf5 file path)
    :param export: bool
    :param export_file_path: str (.hdf5 file path)
    :param label: str
    :param disp: bool
    """
    if config_file_path is not None:
        context.config_file_path = config_file_path
    if 'config_file_path' not in context() or context.config_file_path is None or \
            not os.path.isfile(context.config_file_path):
        raise Exception('nested.optimize: config_file_path specifying required parameters is missing or invalid.')
    config_dict = read_from_yaml(context.config_file_path)
    if 'param_names' not in config_dict or config_dict['param_names'] is None:
        raise Exception('nested.optimize: config_file at path: %s is missing the following required field: %s' %
                        (context.config_file_path, 'param_names'))
    else:
        context.param_names = config_dict['param_names']
    if 'default_params' not in config_dict or config_dict['default_params'] is None:
        context.default_params = {}
    else:
        context.default_params = config_dict['default_params']
    if 'bounds' not in config_dict or config_dict['bounds'] is None:
        raise Exception('nested.optimize: config_file at path: %s is missing the following required field: %s' %
                        (context.config_file_path, 'bounds'))
    for param in context.default_params:
        config_dict['bounds'][param] = (context.default_params[param], context.default_params[param])
    context.bounds = [config_dict['bounds'][key] for key in context.param_names]
    if 'rel_bounds' not in config_dict or config_dict['rel_bounds'] is None:
        context.rel_bounds = None
    else:
        context.rel_bounds = config_dict['rel_bounds']
    if 'x0' not in config_dict or config_dict['x0'] is None:
        context.x0 = None
    else:
        context.x0 = config_dict['x0']
        context.x0_dict = context.x0
        for param_name in context.default_params:
            context.x0_dict[param_name] = context.default_params[param_name]
        context.x0_array = param_dict_to_array(context.x0_dict, context.param_names)

    missing_config = []
    if 'feature_names' not in config_dict or config_dict['feature_names'] is None:
        missing_config.append('feature_names')
    else:
        context.feature_names = config_dict['feature_names']
    if 'objective_names' not in config_dict or config_dict['objective_names'] is None:
        missing_config.append('objective_names')
    else:
        context.objective_names = config_dict['objective_names']
    if 'target_val' in config_dict:
        context.target_val = config_dict['target_val']
    else:
        context.target_val = None
    if 'target_range' in config_dict:
        context.target_range = config_dict['target_range']
    else:
        context.target_range = None
    if 'optimization_title' in config_dict:
        if config_dict['optimization_title'] is None:
            context.optimization_title = ''
        else:
            context.optimization_title = config_dict['optimization_title']
    if 'kwargs' in config_dict and config_dict['kwargs'] is not None:
        context.kwargs = config_dict['kwargs']  # Extra arguments to be passed to imported sources
    else:
        context.kwargs = {}
    context.kwargs.update(kwargs)
    context.update(context.kwargs)

    if 'update_context' not in config_dict or config_dict['update_context'] is None:
        context.update_context_list = []
    else:
        context.update_context_list = config_dict['update_context']
    if 'get_features_stages' not in config_dict or config_dict['get_features_stages'] is None:
        missing_config.append('get_features_stages')
    else:
        context.stages = config_dict['get_features_stages']
    if 'get_objectives' not in config_dict or config_dict['get_objectives'] is None:
        missing_config.append('get_objectives')
    else:
        context.get_objectives_dict = config_dict['get_objectives']
    if missing_config:
        raise Exception('nested.optimize: config_file at path: %s is missing the following required fields: %s' %
                        (context.config_file_path, ', '.join(str(field) for field in missing_config)))

    if label is not None:
        context.label = label
    if 'label' not in context() or context.label is None:
        label = ''
    else:
        label = '_' + context.label

    if output_dir is not None:
        context.output_dir = output_dir
    if 'output_dir' not in context():
        context.output_dir = None
    if context.output_dir is None:
        output_dir_str = ''
    else:
        output_dir_str = context.output_dir + '/'

    if temp_output_path is not None:
        context.temp_output_path = temp_output_path
    if 'temp_output_path' not in context() or context.temp_output_path is None:
        context.temp_output_path = '%s%s_pid%i_%s%s_temp_output.hdf5' % \
                                   (output_dir_str, datetime.datetime.today().strftime('%Y%m%d%H%M'), os.getpid(),
                                    context.optimization_title, label)
    context.export = export
    if export_file_path is not None:
        context.export_file_path = export_file_path
    if 'export_file_path' not in context() or context.export_file_path is None:
        context.export_file_path = '%s%s_%s%s_interactive_exported_output.hdf5' % \
                                   (output_dir_str, datetime.datetime.today().strftime('%Y%m%d%H%M'),
                                    context.optimization_title, label)
    context.disp = disp
    context.rel_bounds_handler = RelativeBoundedStep(context.x0_array, context.param_names, context.bounds,
                                                     context.rel_bounds)

    local_source = os.path.basename(source_file_name).split('.')[0]
    m = sys.modules['__main__']
    context.update_context_funcs = []
    for source, func_name in context.update_context_list:
        if source == local_source:
            try:
                func = getattr(m, func_name)
                if not isinstance(func, collections.Callable):
                    raise Exception('nested.optimize: update_context function: %s not callable' % func_name)
                context.update_context_funcs.append(func)
            except Exception:
                raise ImportError('nested.optimize: update_context function: %s not found' % func_name)
    if not context.update_context_funcs:
        raise ImportError('nested.optimize: update_context function not found')

    if 'comm' not in context():
        try:
            from mpi4py import MPI
            context.comm = MPI.COMM_WORLD
        except Exception:
            print 'ImportWarning: nested.optimize: source: %s; config_interactive: problem importing from mpi4py' % \
                  local_source

    if hasattr(m, 'config_worker'):
        config_func = getattr(m, 'config_worker')
        if not isinstance(config_func, collections.Callable):
            raise Exception('nested.optimize: source: %s; config_interactive: problem executing config_worker' %
                            local_source)
        config_func(context.update_context_funcs, context.param_names, context.default_params, context.feature_names,
                    context.objective_names, context.target_val, context.target_range, context.temp_output_path,
                    context.export_file_path, context.output_dir, context.disp, **context.kwargs)
    update_source_contexts(context.x0_array, context)


def merge_exported_data(file_path_list, new_file_path=None, verbose=True):
    """
    Each nested.optimize worker can export data intermediates to its own unique .hdf5 file (temp_output_path). Then the
    master process collects and merges these files into a single file (export_file_path). To avoid redundancy, this
    method only copies the top-level group 'shared_context' once. Then, the content of any other top-level groups
    are copied recursively. If a group attribute 'enumerated' exists and is True, this method expects data to be nested
    in groups enumerated with str(int) as keys. These data structures will be re-enumerated during the merge. Otherwise,
    groups containing nested data are expected to be labeled with unique keys, and nested structures are only copied
    once.
    :param file_path_list: list of str (paths)
    :param new_file_path: str (path)
    :return str (path)
    """
    if new_file_path is None:
        new_file_path = 'merged_hdf5_'+datetime.datetime.today().strftime('%m%d%Y%H%M')+'_'+os.getpid()
    if not len(file_path_list) > 0:
        return None
    enum = 0
    with h5py.File(new_file_path, 'a') as new_f:
        for old_file_path in file_path_list:
            with h5py.File(old_file_path, 'r') as old_f:
                for group in old_f:
                    if group == 'shared_context':
                        if group not in new_f:
                            new_f.copy(old_f[group], new_f)
                    else:
                        if group not in new_f:
                            new_f.create_group(group)
                        target = new_f[group]
                        if 'enumerated' in old_f[group].attrs and old_f[group].attrs['enumerated']:
                            enumerated = True
                        else:
                            enumerated = False
                        target.attrs['enumerated'] = enumerated
                        if enumerated:
                            if verbose:
                                print 'enumerated', group, old_f[group], target
                            for source in old_f[group].itervalues():
                                target.copy(source, target, name=str(enum))
                                enum += 1
                        else:
                            if verbose:
                                print 'not enumerated', group, old_f[group], target
                            h5_nested_copy(old_f[group], target)
    if verbose:
        print 'merge_hdf5_files: exported to file_path: %s' % new_file_path
    return new_file_path


def h5_nested_copy(source, target):
    """

    :param source: :class: in ['h5py.File', 'h5py.Group', 'h5py.Dataset']
    :param target: :class: in ['h5py.File', 'h5py.Group']
    """
    if isinstance(source, h5py.Dataset):
        try:
            target.copy(source, target)
        except IOError:
            pass
        return
    else:
        for key, val in source.iteritems():
            if key in target:
                h5_nested_copy(val, target[key])
            else:
                target.copy(val, target, name=key)


def update_source_contexts(x, local_context=None):
    """

    :param x: array
    :param local_context: :class:'Context'
    """
    if local_context is None:
        module = sys.modules['__main__']
        for item_name in dir(module):
            if isinstance(getattr(module, item_name), Context):
                local_context = getattr(module, item_name)
    if hasattr(local_context, 'update_context_funcs'):
        local_context.x_array = x
        for update_func in local_context.update_context_funcs:
            update_func(x, local_context)


"""
--------------------------functions to plot local sensitivity-------------------
to call the function:
    from nested.optimize_utils import * 
    pop = PopulationStorage(file_path='path_to_hdf5_file.hdf5')
    local_sensitivity(pop)
"""


def pop_to_matrix(population):
    """
    converts collection of individuals in PopulationStorage into a matrix for data manipulation

    :param population: PopulationStorage object
    :return: data, 2d array. rows = each data point or individual, col = parameters, then objectives
    """
    data = []
    generation_array = population.history
    for generation in generation_array:
        for datum in generation:
            x_array = datum.x
            objectives_array = datum.objectives
            individual_array = np.append(x_array, objectives_array, axis=0)

            data.append(individual_array)

    data = np.array(data)
    return data


def best_to_array(population):
    best_individual = population.get_best()[0]
    best_array = best_individual.x
    best_obj_array = best_individual.objectives
    best_array = np.append(best_array, best_obj_array, axis=0)
    num_parameters = len(best_individual.x)

    return best_array, num_parameters


def get_linear_arrays(data):
    min_array = np.min(data, axis=0)
    max_array = np.max(data, axis=0)
    diff_array = abs(max_array - min_array)

    return min_array, diff_array


def get_log_arrays(data_log_10):
    logmin_array = np.min(data_log_10, axis=0)
    logmin_array[np.isnan(logmin_array)] = 0
    logmax_array = np.max(data_log_10, axis=0)
    logmax_array[np.isnan(logmax_array)] = 0
    logdiff_array = abs(logmax_array - logmin_array)

    return logmin_array, logdiff_array


def normalize_data(population, data):
    """
    normalize all data points. used for calculating neighborship

    :param population: PopulationStorage object
    :param data: 2d array object with data from generations
    :return: matrix of normalized values for parameters and objectives
    """
    best_array, num_parameters = best_to_array(population)
    data_normed = np.copy(data)
    best_normed = np.copy(best_array)
    best_log = np.log10(np.copy(best_array))

    num_rows, num_cols = data.shape
    min_array, diff_array = get_linear_arrays(data)

    data_log_10 = np.log10(np.copy(data))
    logmin_array, logdiff_array = get_log_arrays(data_log_10)

    scaling = []  # holds a list of whether the column was log or lin normalized

    # iterate column-wise over each parameter and objective.
    # if the magnitude of the range is greater than 2, log normalization. otherwise, linear norm
    for i in range(num_cols):
        if logdiff_array[i] < 2:  # lin
            min_vector = np.full((num_rows,), min_array[i])
            diff_vector = np.full((num_rows,), diff_array[i])

            data_normed[:, i] = np.true_divide((data[:, i] - min_vector), diff_vector)
            best_normed[i] = np.true_divide((best_normed[i] - min_array[i]), diff_array[i])
            scaling.append('lin')
        else:  # log
            logmin_vector = np.full((num_rows,), logmin_array[i])
            logdiff_vector = np.full((num_rows,), logdiff_array[i])

            data_normed[:, i] = np.true_divide((data_log_10[:, i] - logmin_vector), logdiff_vector)
            best_normed[i] = np.true_divide((best_log[i] - logmin_array[i]), logdiff_array[i])
            scaling.append('log')

    data_normed = np.nan_to_num(data_normed)
    best_normed = np.array(np.nan_to_num(best_normed))

    X_normed = data_normed[:, 0:num_parameters]
    y_normed = data_normed[:, num_parameters:]

    print("Data normalized")
    return X_normed, y_normed, best_normed


def get_important_parameters(data, num_parameters, num_objectives, feature_names):
    """
    using decision trees, get important parameters for each objective.

    :param data: 2d array
    :param num_parameters: int
    :param num_objectives: int
    :param feature_names: list of strings
    :return: important parameters - a list of lists. list length = num_objectives
    """
    # the sum of feature_importances_ is 1, so the baseline should be relative to num_params
    # the below calculation is pretty ad hoc and based on personal observations
    feat_baseline = .1 - ((num_parameters % 500) - 20) / 500

    X = data[:, 0:num_parameters]
    y = data[:, num_parameters:]
    important_parameters = [[] for x in range(num_objectives)]

    # create a decision tree for each objective. feature is considered "important" if over the baseline
    for i in range(num_objectives):
        dt = DecisionTreeRegressor(random_state=0)
        dt.fit(X, y[:, i])

        param_list = list(zip(map(lambda t: round(t, 4), dt.feature_importances_), feature_names))
        for j in range(len(dt.feature_importances_)):
            if dt.feature_importances_[j] > feat_baseline:
                important_parameters[i].append(param_list[j][1])

    print("Important parameters calculated")
    print(important_parameters)
    return important_parameters


def split_parameters(num_parameters, important_parameters_set, param_names, p):
    # get important parameters for the objective
    feature_indices = []
    if important_parameters_set:
        for j in range(len(important_parameters_set)):  # hm
            index = np.where(param_names == important_parameters_set[j])[0][0]
            feature_indices.append(index)
    else:
        return [], [x for x in range(num_parameters)], []

    # create subsets of the parameter matrix based on importance
    important = [x for x in feature_indices if x != p]
    unimportant = [x for x in range(num_parameters) if x not in important and x != p]

    return important, unimportant, feature_indices


def possible_neighbors(important, unimportant, X_normed, X_best_normed, max_dist,
                       unimportant_distance, counter, magnitude):

    # get second set of neighbors (filter important params)
    if important:
        important_cheb_tree = BallTree(X_normed[:, important], metric='chebyshev')
        important_neighbor_array = important_cheb_tree.query_radius(X_best_normed[important].reshape(1, -1),
                                                                    r=max_dist + 10 ** magnitude * counter)
    else:
        important_neighbor_array = []

    if unimportant:
        # get neighbors (filter unimportant parameters)
        unimportant_cheb_tree = BallTree(X_normed[:, unimportant], metric='chebyshev')
        unimportant_neighbor_array = \
            unimportant_cheb_tree.query_radius(X_best_normed[unimportant].reshape(1, -1),
                                               r=unimportant_distance + 10 ** magnitude * counter * 1.5)
    else:
        unimportant_neighbor_array = important_neighbor_array

    return unimportant_neighbor_array, important_neighbor_array


def get_neighbors(num_parameters, num_objectives, important_parameters, param_names, objective_names, X_normed,
                  best_normed, verbose, n_neighbors, max_dist, unimportant_distance, param_radius):
    """
    get neighbors for each objective/parameter pair based on 1) a max radius for important features
    and 2) a max radius for unimportant features (euclidean distance)

    :param num_parameters: int
    :param num_objectives: int
    :param important_parameters: list of lists of strings
    :param param_names: list of strings
    :param objective_names: list of strings
    :param X_normed: 2d array
    :param best_normed: 1d array
    :param verbose: bool. print statements if true
    :return: neighbor matrix, 2d array with each cell a list of integers (integers = neighbor indices in data matrix)
    """
    X_best_normed = best_normed[0:num_parameters]
    neighbor_matrix = np.empty((num_parameters, num_objectives), dtype=object)
    x_not = np.where(X_normed == X_best_normed)[0][0]
    magnitude = int(math.log10(max_dist))

    for p in range(num_parameters):  # row
        for o in range(num_objectives):  # col
            counter = 1
            while counter == 1 or len(neighbor_matrix[p][o]) < n_neighbors:
                if max_dist * counter > .5:
                    print "\nParameter:", param_names[p], "/ Objective:", objective_names[o], ": Neighbors not " \
                          "found for specified n_neighbor threshold"
                    break

                # get important vs unimportant parameters
                important, unimportant, feature_indices = \
                    split_parameters(num_parameters, important_parameters[o], param_names, p)

                #if not important and not feature_indices:
                #    print "\nParameter:", param_names[p], "/ Objective:", objective_names[o], "\nSKIPPED because "\
                #          "no strongly important parameters were identified for this objective"
                #    break

                # get neighbor arrays based on important param distance and unimportant param distance
                unimportant_neighbor_array, important_neighbor_array = possible_neighbors(important, unimportant,
                                        X_normed, X_best_normed, max_dist, unimportant_distance, counter, magnitude)

                # filter according to the above constraints and if query parameter perturbation > twice
                # the max perturbation of unimportant parameters
                filtered_neighbors = [x_not]
                num_neighbors = len(unimportant_neighbor_array[0])
                for k in range(num_neighbors):
                    point_index = int(unimportant_neighbor_array[0][k])
                    significant_perturbation = abs(X_normed[point_index, p] - X_best_normed[p]) > 2 * max_dist
                    local = abs(X_normed[point_index, p] - X_best_normed[p]) <= param_radius
                    if significant_perturbation and local and (not important or point_index in important_neighbor_array[0]):
                        filtered_neighbors.append(point_index)

                if len(filtered_neighbors) >= n_neighbors and verbose:
                    print "\nParameter:", param_names[p], "/ Objective:", objective_names[o]
                    print "Max distance (for important parameters):", max_dist + 10**magnitude*counter
                    print "Neighbors:", len(filtered_neighbors)

                neighbor_matrix[p][o] = filtered_neighbors
                counter = counter + 1
    return neighbor_matrix


def get_coef(num_parameters, num_objectives, neighbor_matrix, X_normed, y_normed):
    """
    compute coefficients between parameter and objective based on linear regression. also get p-val
    coef will always refer to the beta coefficient/slope in linear regression between param X and objective y

    :param num_parameters: int
    :param num_objectives: int
    :param neighbor_matrix: 2d array of lists which contain neighbor indices
    :param X_normed: 2d array of parameters normalized
    :param y_normed: 2d array of objectives normalized
    :return:
    """
    coef_matrix = np.zeros((num_parameters, num_objectives))
    pearson_matrix = np.ones((num_parameters, num_objectives))

    for param in range(num_parameters):
        for obj in range(num_objectives):
            neighbor_array = neighbor_matrix[param][obj]
            if neighbor_array:
                num_neighbors = len(neighbor_array)
                selection = [ind for ind in neighbor_array]
                X_sub = X_normed[selection, param]  # get relevant X data points

                regr = LinearRegression()
                regr.fit(X_sub.reshape(num_neighbors, 1), y_normed[selection, obj].reshape(num_neighbors, 1))

                coef_matrix[param][obj] = regr.coef_
                pearson_matrix[param][obj] = (pearsonr(X_sub, y_normed[selection, obj]))[1]
    return coef_matrix, pearson_matrix


def normalize_coef(num_parameters, num_objectives, coef_matrix, pearson_matrix, p_baseline):
    """
    normalize absolute beta coefficients (regression slope) by column. only normalize the ones
    less than the pval

    :param num_parameters: int
    :param num_objectives: int
    :param coef_matrix: 2d array (beta coef)
    :param pearson_matrix: 2d array
    :param p_baseline: float between 0 and 1
    :return:
    """
    coef_normed = abs(np.copy(coef_matrix))
    for obj in range(num_objectives):
        sig_values = []
        for param in range(num_parameters):
            if pearson_matrix[param][obj] < p_baseline:
                sig_values.append(abs(coef_matrix[param][obj]))
        if sig_values:  # if no significant values for an objective, they won't be plotted anyway
            max_coef = np.amax(sig_values)
            min_coef = np.amin(sig_values)
            range_coef = max_coef - min_coef

            if range_coef == 0:
                coef_normed[:, obj] = np.full((num_parameters, ), 1)
            else:
                min_vector = np.full((num_parameters,), min_coef)
                range_vector = np.full((num_parameters,), range_coef)

                coef_normed[:, obj] = np.true_divide((coef_normed[:, obj] - min_vector), range_vector)

    return coef_normed


def plot_sensitivity(num_parameters, num_objectives, coef_matrix, pearson_matrix, param_names, objective_names,
                     p_baseline):
    """
    plot local sensitivity. mask cells with p-vals greater than than baseline

    :param num_parameters: int
    :param num_objectives: int
    :param coef_matrix: 2d array of floats
    :param pearson_matrix: 2d array of floats
    :param param_names: list of str
    :param objective_names: list of str
    :param p_baseline: float from 0 to 1
    :return:
    """
    coef_normed = normalize_coef(num_parameters, num_objectives, coef_matrix, pearson_matrix, p_baseline)

    # create mask
    mask = np.full((num_parameters, num_objectives), True, dtype=bool)  # mask
    mask[pearson_matrix < p_baseline] = False  # do not mask

    plt.figure(figsize=(16, 5))
    hm = sns.heatmap(coef_normed, fmt="g", cmap='plasma', vmax=1, vmin=0, mask=mask, linewidths=1)
    hm.set_xticklabels(objective_names)
    hm.set_yticklabels(param_names)
    plt.xticks(rotation=-90)
    plt.yticks(rotation=0)
    plt.title("Absolute Beta Coefficients (Normalized by column)")
    plt.show()


def prompt_values():
    n_neighbors = 30
    alpha_value = .05
    max_dist = .001
    unimportant_distance = .002
    param_radius = .02

    user_input = raw_input('Do you want to specify the values for neighbor search? The default '
                           'values are num neighbors = 30, alpha value = .05, query parameter radius = .02, '
                           'starting radius for important parameters = .001, and unimportant parameters = .002. '
                           '(y/n) ')
    if user_input in ['y', 'Y']:
        n_neighbors = int(raw_input('Threshold for number of neighbors?: '))
        alpha_value = float(raw_input('Alpha value?: '))
        param_radius = float(raw_input('Query parameter radius?: '))
        max_dist = float(raw_input('Starting radius for important parameters?: '))
        unimportant_distance = float(raw_input('Starting radius for unimportant parameters?: '))
    elif user_input in ['n', 'N']:
        print 'Thanks.'
    else:
        while user_input not in ['y', 'Y', 'n', 'N']:
            user_input = raw_input('Please enter y or n. ')

    return n_neighbors, alpha_value, max_dist, unimportant_distance, param_radius


def prompt_neighbor_dialog(num_parameters, num_objectives, important_parameters, param_names, objective_names,
                         X_normed, best_normed, verbose, n_neighbors, max_dist, unimportant_distance, param_radius):
    unacceptable = True
    while unacceptable:
        neighbor_matrix = get_neighbors(num_parameters, num_objectives, important_parameters, param_names,
                                        objective_names, X_normed, best_normed, verbose, n_neighbors, max_dist,
                                        unimportant_distance, param_radius)
        user_input = raw_input('Was this an acceptable outcome (y/n)? ')
        if user_input in ['y', 'Y']:
            unacceptable = False
        elif user_input in ['n', 'N']:
            n_neighbors, alpha_value, max_dist, unimportant_distance = prompt_values()
        else:
            while user_input not in ['y', 'Y', 'n', 'N']:
                user_input = raw_input('Was this an acceptable outcome (y/n)? ')
    return neighbor_matrix


def local_sensitivity(population, verbose=True):
    """
    main function for plotting and computing local sensitivity

    :param population: PopulationStorage object
    :param verbose: bool. if True, will print radius and num neighbors for each parameter/objective pair
    :return:
    """
    data = pop_to_matrix(population)
    X_normed, y_normed, best_normed = normalize_data(population, data)

    param_names = population.param_names
    objective_names = population.objective_names
    num_parameters = len(param_names)
    num_objectives = len(objective_names)

    important_parameters = get_important_parameters(data, num_parameters, num_objectives, param_names)

    n_neighbors, alpha_value, max_dist, unimportant_distance, param_radius = prompt_values()
    neighbor_matrix = prompt_neighbor_dialog(num_parameters, num_objectives, important_parameters, param_names,
                                             objective_names, X_normed, best_normed, verbose, n_neighbors, max_dist,
                                             unimportant_distance, param_radius)

    coef_matrix, pearson_matrix = get_coef(num_parameters, num_objectives, neighbor_matrix, X_normed, y_normed)
    plot_sensitivity(num_parameters, num_objectives, coef_matrix, pearson_matrix, param_names, objective_names,
                     alpha_value)

