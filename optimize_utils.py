"""
Library of functions and classes to support nested.optimize
"""
__author__ = 'Aaron D. Milstein and Grace Ng'
from nested.utils import *
from nested.parallel import find_context, find_context_name
import collections
from scipy._lib._util import check_random_state
from copy import deepcopy

import numpy as np
from collections import defaultdict
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import BallTree
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy import stats
import matplotlib.pyplot as plt
import math
import warnings


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
            # Enable tracking of user-defined attributes through kwargs to 'append'
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
            print('PopulationStorage: Defaulting to get_best in last iteration.')
        elif type(iterations) == int and iterations * self.path_length > len(self.history):
            iterations = 'all'
            print('PopulationStorage: Defaulting to get_best across all iterations.')
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
        elif isinstance(evaluate, basestring) and evaluate in globals() and isinstance(globals()[evaluate], collections.Callable):
            evaluate_name = evaluate
            evaluate = globals()[evaluate_name]
        else:
            raise TypeError('PopulationStorage: evaluate must be callable.')
        if modify:
            group = [individual for population in self.history[start:end] for individual in population]
            if start > 0:
                group.extend([individual for individual in self.survivors[start - 1]])
        else:
            group = [deepcopy(individual) for population in self.history[start:end] for individual in population]
            if start > 0:
                group.extend([deepcopy(individual) for individual in self.survivors[start - 1]])
        evaluate(group, self.min_objectives, self.max_objectives)
        group = sort_by_rank(group)
        if n == 'all':
            return group
        else:
            return group[:n]

    def plot(self, subset=None, show_failed=False):
        """

        :param subset: can be str, list, or dict
            valid categories: 'features', 'objectives', 'parameters'
            valid dict vals: list of str of valid category names
        :param show_failed: bool; whether to show failed models when plotting parameters
        """
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import cm
        import matplotlib as mpl
        from matplotlib.lines import Line2D
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        mpl.rcParams['svg.fonttype'] = 'none'
        mpl.rcParams['text.usetex'] = False
        cmap = cm.rainbow

        default_categories = {'parameters': self.param_names, 'objectives': self.objective_names,
                              'features': self.feature_names}
        if subset is None:
            categories = default_categories
        elif isinstance(subset, basestring):
            if subset not in default_categories:
                raise KeyError('PopulationStorage.plot: invalid category provided to subset argument: %s' % subset)
            else:
                categories = {subset: default_categories[subset]}
        elif isinstance(subset, list):
            categories = dict()
            for key in subset:
                if key not in default_categories:
                    raise KeyError('PopulationStorage.plot: invalid category provided to subset argument: %s' % key)
                categories[key] = default_categories[key]
        elif isinstance(subset, dict):
            for key in subset:
                if key not in default_categories:
                    raise KeyError('PopulationStorage.plot: invalid category provided to subset argument: %s' % key)
                if not isinstance(subset[key], list):
                    raise ValueError('PopulationStorage.plot: subset category names must be provided as a list')
                valid_elements = default_categories[key]
                for element in subset[key]:
                    if element not in valid_elements:
                        raise KeyError('PopulationStorage.plot: invalid %s name provided to subset argument: %s' %
                                       (key[:-1], element))
            categories = subset
        else:
            raise ValueError('PopulationStorage.plot: invalid type of subset argument')

        fig, axes = plt.subplots(1, figsize=(6.5, 4.8))
        all_ranks_history = []
        all_fitness_history = []
        survivor_ranks_history = []
        survivor_fitness_history = []
        for j, population in enumerate(self.history):
            if j % self.path_length == 0:
                this_all_ranks_pop = []
                this_all_fitness_pop = []
                this_survivor_ranks_pop = []
                this_survivor_fitness_pop = []
            for indiv in population:
                this_all_ranks_pop.append(indiv.rank)
                this_all_fitness_pop.append(indiv.fitness)
            if (j + 1) % self.path_length == 0:
                all_ranks_history.append(this_all_ranks_pop)
                all_fitness_history.append(this_all_fitness_pop)
                for indiv in self.survivors[j]:
                    this_survivor_ranks_pop.append(indiv.rank)
                    this_survivor_fitness_pop.append(indiv.fitness)
                survivor_ranks_history.append(this_survivor_ranks_pop)
                survivor_fitness_history.append(this_survivor_fitness_pop)

        max_fitness = float(max(np.max(all_fitness_history, axis=0)))
        norm = mpl.colors.Normalize(vmin=-0.5, vmax=max_fitness + 0.5)
        for i in range(len(all_ranks_history)):
            this_colors = list(cmap(np.divide(all_fitness_history[i], max_fitness)))
            axes.scatter(np.ones(len(all_ranks_history[i])) * i, all_ranks_history[i], c=this_colors,
                         alpha=0.2, s=5., linewidth=0)
            this_colors = list(cmap(np.divide(survivor_fitness_history[i], max_fitness)))
            axes.scatter(np.ones(len(survivor_ranks_history[i])) * i, survivor_ranks_history[i], c=this_colors,
                         alpha=0.4, s=10., linewidth=0.5, edgecolor='k')
        axes.set_xlabel('Number of iterations')
        axes.set_ylabel('Model rank')
        axes.set_title('Fitness')
        divider = make_axes_locatable(axes)
        cax = divider.append_axes('right', size='3%', pad=0.1)
        cbar = mpl.colorbar.ColorbarBase(cax, cmap=cm.get_cmap('rainbow', int(max_fitness + 1)), norm=norm,
                                         orientation='vertical')
        cbar.set_label('Fitness', rotation=-90)
        cbar.set_ticks(list(range(int(max_fitness + 1))))
        cbar.ax.get_yaxis().labelpad = 15
        clean_axes(axes)
        fig.show()

        fig, axes = plt.subplots(1, figsize=(7., 4.8))
        all_rel_energy_history = []
        survivor_rel_energy_history = []
        for j, population in enumerate(self.history):
            if j % self.path_length == 0:
                this_all_rel_energy_pop = []
                this_survivor_rel_energy_pop = []
            for indiv in population:
                this_all_rel_energy_pop.append(indiv.energy)
            if (j + 1) % self.path_length == 0:
                all_rel_energy_history.append(this_all_rel_energy_pop)
                for indiv in self.survivors[j]:
                    this_survivor_rel_energy_pop.append(indiv.energy)
                survivor_rel_energy_history.append(this_survivor_rel_energy_pop)

        iterations = list(range(len(all_rel_energy_history)))
        all_rel_energy_mean = np.array([np.mean(all_rel_energy_history[i]) for i in range(len(iterations))])
        all_rel_energy_med = np.array([np.median(all_rel_energy_history[i]) for i in range(len(iterations))])
        all_rel_energy_std = np.array([np.std(all_rel_energy_history[i]) for i in range(len(iterations))])

        for i in iterations:
            axes.scatter(np.ones(len(all_rel_energy_history[i])) * i, all_rel_energy_history[i], c='none',
                         edgecolor='salmon', linewidth=0.5, alpha=0.2, s=5.)
            axes.scatter(np.ones(len(survivor_rel_energy_history[i])) * i, survivor_rel_energy_history[i], c='none',
                         edgecolor='k', linewidth=0.5, alpha=0.4, s=10.)
        axes.plot(iterations, all_rel_energy_med, c='r')
        axes.fill_between(iterations, all_rel_energy_mean - all_rel_energy_std,
                          all_rel_energy_mean + all_rel_energy_std, alpha=0.35, color='salmon')
        legend_elements = [Line2D([0], [0], marker='o', color='salmon', label='All models', markerfacecolor='none',
                                  markersize=5, markeredgewidth=1.5, linewidth=0),
                           Line2D([0], [0], marker='o', color='k', label='Survivors', markerfacecolor='none',
                                  markersize=5, markeredgewidth=1.5, linewidth=0),
                           Line2D([0], [0], color='r', lw=2, label='Median')]
        axes.set_xlabel('Number of iterations')
        axes.set_ylabel('Multi-objective error score')
        axes.set_title('Multi-objective error score')
        axes.legend(handles=legend_elements, loc='center', frameon=False, handlelength=1, bbox_to_anchor=(1.1, 0.5))
        clean_axes(axes)
        fig.subplots_adjust(right=0.8)
        fig.show()

        fig, axes = plt.subplots(1, figsize=(7., 4.8))
        all_abs_energy_history = []
        survivor_abs_energy_history = []
        for j, population in enumerate(self.history):
            if j % self.path_length == 0:
                this_all_abs_energy_pop = []
                this_survivor_abs_energy_pop = []
            for indiv in population:
                this_all_abs_energy_pop.append(np.sum(indiv.objectives))
            if (j + 1) % self.path_length == 0:
                all_abs_energy_history.append(this_all_abs_energy_pop)
                for indiv in self.survivors[j]:
                    this_survivor_abs_energy_pop.append(np.sum(indiv.objectives))
                survivor_abs_energy_history.append(this_survivor_abs_energy_pop)

        iterations = list(range(len(all_abs_energy_history)))
        all_abs_energy_mean = np.array([np.mean(all_abs_energy_history[i]) for i in range(len(iterations))])
        all_abs_energy_med = np.array([np.median(all_abs_energy_history[i]) for i in range(len(iterations))])
        all_abs_energy_std = np.array([np.std(all_abs_energy_history[i]) for i in range(len(iterations))])

        for i in iterations:
            axes.scatter(np.ones(len(all_abs_energy_history[i])) * i, all_abs_energy_history[i], c='none',
                         edgecolor='salmon', linewidth=0.5, alpha=0.2, s=5.)
            axes.scatter(np.ones(len(survivor_abs_energy_history[i])) * i, survivor_abs_energy_history[i], c='none',
                         edgecolor='k', linewidth=0.5, alpha=0.4, s=10.)
        # axes.set_yscale('log')
        axes.plot(iterations, all_abs_energy_med, c='r')
        axes.fill_between(iterations, all_abs_energy_mean - all_abs_energy_std,
                          all_abs_energy_mean + all_abs_energy_std, alpha=0.35, color='salmon')
        legend_elements = [Line2D([0], [0], marker='o', color='salmon', label='All models', markerfacecolor='none',
                                  markersize=5, markeredgewidth=1.5, linewidth=0),
                           Line2D([0], [0], marker='o', color='k', label='Survivors', markerfacecolor='none',
                                  markersize=5, markeredgewidth=1.5, linewidth=0),
                           Line2D([0], [0], color='r', lw=2, label='Median')]
        axes.set_xlabel('Number of iterations')
        # axes.set_ylabel('Total objective error (log scale)')
        axes.set_ylabel('Total objective error')
        axes.set_title('Total objective error')
        axes.legend(handles=legend_elements, loc='center', frameon=False, handlelength=1, bbox_to_anchor=(1.1, 0.5))
        clean_axes(axes)
        fig.subplots_adjust(right=0.8)
        fig.show()

        if 'parameters' in categories:
            name_list = self.param_names.tolist()
            for param_name in categories['parameters']:
                index = name_list.index(param_name)
                fig, axes = plt.subplots(1, figsize=(7., 4.8))
                all_param_history = []
                failed_param_history = []
                survivor_param_history = []
                for j, population in enumerate(self.history):
                    if j % self.path_length == 0:
                        this_all_param_pop = []
                        this_failed_param_pop = []
                        this_survivor_param_pop = []
                    for indiv in population:
                        this_all_param_pop.append(indiv.x[index])
                    if len(self.failed[j]) > 0:
                        for indiv in self.failed[j]:
                            this_failed_param_pop.append(indiv.x[index])
                    if (j + 1) % self.path_length == 0:
                        all_param_history.append(this_all_param_pop)
                        for indiv in self.survivors[j]:
                            this_survivor_param_pop.append(indiv.x[index])
                        survivor_param_history.append(this_survivor_param_pop)
                        failed_param_history.append(this_failed_param_pop)

                iterations = list(range(len(all_param_history)))
                all_param_mean = np.array([np.mean(all_param_history[i]) for i in range(len(iterations))])
                all_param_med = np.array([np.median(all_param_history[i]) for i in range(len(iterations))])
                all_param_std = np.array([np.std(all_param_history[i]) for i in range(len(iterations))])

                for i in iterations:
                    axes.scatter(np.ones(len(all_param_history[i])) * i, all_param_history[i], c='none',
                                 edgecolor='salmon', linewidth=0.5, alpha=0.2, s=5.)
                    if show_failed:
                        axes.scatter(np.ones(len(failed_param_history[i])) * (i + 0.5), failed_param_history[i],
                                     c='grey', linewidth=0, alpha=0.2, s=5.)
                    axes.scatter(np.ones(len(survivor_param_history[i])) * i, survivor_param_history[i], c='none',
                                 edgecolor='k', linewidth=0.5, alpha=0.4, s=10.)
                axes.plot(iterations, all_param_med, c='r')
                axes.fill_between(iterations, all_param_mean - all_param_std,
                                  all_param_mean + all_param_std, alpha=0.35, color='salmon')
                legend_elements = [Line2D([0], [0], marker='o', color='salmon', label='All models',
                                          markerfacecolor='none', markersize=5, markeredgewidth=1.5, linewidth=0)]
                if show_failed:
                    legend_elements.append(Line2D([0], [0], marker='o', color='none', label='Failed models',
                                                  markerfacecolor='grey', markersize=5, markeredgewidth=0, linewidth=0))
                legend_elements.extend([
                    Line2D([0], [0], marker='o', color='k', label='Survivors', markerfacecolor='none',
                           markersize=5, markeredgewidth=1.5, linewidth=0),
                    Line2D([0], [0], color='r', lw=2, label='Median')])
                axes.set_xlabel('Number of iterations')
                axes.set_ylabel('Parameter value')
                axes.set_title('Parameter: %s' % param_name)
                axes.legend(handles=legend_elements, loc='center', frameon=False, handlelength=1,
                            bbox_to_anchor=(1.1, 0.5))
                clean_axes(axes)
                fig.subplots_adjust(right=0.8)
                fig.show()

        if 'features' in categories:
            name_list = self.feature_names.tolist()
            for feature_name in categories['features']:
                index = name_list.index(feature_name)
                fig, axes = plt.subplots(1, figsize=(7., 4.8))
                all_feature_history = []
                survivor_feature_history = []
                for j, population in enumerate(self.history):
                    if j % self.path_length == 0:
                        this_all_feature_pop = []
                        this_survivor_feature_pop = []
                    for indiv in population:
                        this_all_feature_pop.append(indiv.features[index])
                    if (j + 1) % self.path_length == 0:
                        all_feature_history.append(this_all_feature_pop)
                        for indiv in self.survivors[j]:
                            this_survivor_feature_pop.append(indiv.features[index])
                        survivor_feature_history.append(this_survivor_feature_pop)

                iterations = list(range(len(all_feature_history)))
                all_feature_mean = np.array([np.mean(all_feature_history[i]) for i in range(len(iterations))])
                all_feature_med = np.array([np.median(all_feature_history[i]) for i in range(len(iterations))])
                all_feature_std = np.array([np.std(all_feature_history[i]) for i in range(len(iterations))])

                for i in iterations:
                    axes.scatter(np.ones(len(all_feature_history[i])) * i, all_feature_history[i], c='none',
                                 edgecolor='salmon', linewidth=0.5, alpha=0.2, s=5.)
                    axes.scatter(np.ones(len(survivor_feature_history[i])) * i, survivor_feature_history[i],
                                 c='none', edgecolor='k', linewidth=0.5, alpha=0.4, s=10.)
                axes.plot(iterations, all_feature_med, c='r')
                axes.fill_between(iterations, all_feature_mean - all_feature_std,
                                  all_feature_mean + all_feature_std, alpha=0.35, color='salmon')
                legend_elements = [
                    Line2D([0], [0], marker='o', color='salmon', label='All models', markerfacecolor='none',
                           markersize=5, markeredgewidth=1.5, linewidth=0),
                    Line2D([0], [0], marker='o', color='k', label='Survivors', markerfacecolor='none',
                           markersize=5, markeredgewidth=1.5, linewidth=0),
                    Line2D([0], [0], color='r', lw=2, label='Median')]
                axes.set_xlabel('Number of iterations')
                axes.set_ylabel('Feature value')
                axes.set_title('Feature: %s' % feature_name)
                axes.legend(handles=legend_elements, loc='center', frameon=False, handlelength=1,
                            bbox_to_anchor=(1.1, 0.5))
                clean_axes(axes)
                fig.subplots_adjust(right=0.8)
                fig.show()

        if 'objectives' in categories:
            name_list = self.objective_names.tolist()
            for objective_name in categories['objectives']:
                index = name_list.index(objective_name)
                fig, axes = plt.subplots(1, figsize=(7., 4.8))
                all_objective_history = []
                survivor_objective_history = []
                for j, population in enumerate(self.history):
                    if j % self.path_length == 0:
                        this_all_objective_pop = []
                        this_survivor_objective_pop = []
                    for indiv in population:
                        this_all_objective_pop.append(indiv.objectives[index])
                    if (j + 1) % self.path_length == 0:
                        all_objective_history.append(this_all_objective_pop)
                        for indiv in self.survivors[j]:
                            this_survivor_objective_pop.append(indiv.objectives[index])
                        survivor_objective_history.append(this_survivor_objective_pop)

                iterations = list(range(len(all_objective_history)))
                all_objective_mean = np.array([np.mean(all_objective_history[i]) for i in range(len(iterations))])
                all_objective_med = np.array([np.median(all_objective_history[i]) for i in range(len(iterations))])
                all_objective_std = np.array([np.std(all_objective_history[i]) for i in range(len(iterations))])

                for i in iterations:
                    axes.scatter(np.ones(len(all_objective_history[i])) * i, all_objective_history[i], c='none',
                                 edgecolor='salmon', linewidth=0.5, alpha=0.2, s=5.)
                    axes.scatter(np.ones(len(survivor_objective_history[i])) * i, survivor_objective_history[i],
                                 c='none', edgecolor='k', linewidth=0.5, alpha=0.4, s=10.)
                axes.plot(iterations, all_objective_med, c='r')
                axes.fill_between(iterations, all_objective_mean - all_objective_std,
                                  all_objective_mean + all_objective_std, alpha=0.35, color='salmon')
                legend_elements = [
                    Line2D([0], [0], marker='o', color='salmon', label='All models', markerfacecolor='none',
                           markersize=5, markeredgewidth=1.5, linewidth=0),
                    Line2D([0], [0], marker='o', color='k', label='Survivors', markerfacecolor='none',
                           markersize=5, markeredgewidth=1.5, linewidth=0),
                    Line2D([0], [0], color='r', lw=2, label='Median')]
                axes.set_xlabel('Number of iterations')
                # axes.set_yscale('log')
                axes.set_ylabel('Objective error')
                # axes.set_ylabel('Objective error (log scale)')
                axes.set_title('Objective: %s' % objective_name)
                axes.legend(handles=legend_elements, loc='center', frameon=False, handlelength=1,
                            bbox_to_anchor=(1.1, 0.5))
                clean_axes(axes)
                fig.subplots_adjust(right=0.8)
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
            if 'param_names' not in f.attrs:
                set_h5py_attr(f.attrs, 'param_names', self.param_names)
            if 'feature_names' not in f.attrs:
                set_h5py_attr(f.attrs, 'feature_names', self.feature_names)
            if 'objective_names' not in f.attrs:
                set_h5py_attr(f.attrs, 'objective_names', self.objective_names)
            if 'path_length' not in f.attrs:
                f.attrs['path_length'] = self.path_length
            if n is None:
                n = 1
            elif n == 'all':
                n = len(self.history)
            elif not isinstance(n, int):
                n = 1
                print('PopulationStorage: defaulting to exporting last generation to file.')
            gen_index = len(self.history) - n
            j = n
            while n > 0:
                if str(gen_index) in f:
                    print('PopulationStorage: generation %s already exported to file.')
                else:
                    f.create_group(str(gen_index))
                    for key in self.attributes:
                        set_h5py_attr(f[str(gen_index)].attrs, key, self.attributes[key][gen_index])
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
                                                                                     compression='gzip')
                                f[str(gen_index)][group_name][str(i)].create_dataset('objectives',
                                                                                     data=[self.None2nan(val) for val in
                                                                                           individual.objectives],
                                                                                     compression='gzip')
                            f[str(gen_index)][group_name][str(i)].create_dataset('x',
                                                                                 data=[self.None2nan(val) for val in
                                                                                       individual.x],
                                                                                 compression='gzip')
                n -= 1
                gen_index += 1
        print('PopulationStorage: saved %i generations (up to generation %i) to file: %s' % (
        j, gen_index - 1, file_path))

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
            self.param_names = get_h5py_attr(f.attrs, 'param_names')
            self.feature_names = get_h5py_attr(f.attrs, 'feature_names')
            self.objective_names = get_h5py_attr(f.attrs, 'objective_names')
            self.path_length = f.attrs['path_length']
            for gen_index in range(len(f)):
                for key in f[str(gen_index)].attrs:
                    if key not in self.attributes:
                        self.attributes[key] = []
                    self.attributes[key].append(get_h5py_attr(f[str(gen_index)].attrs, key))
                history, survivors, failed = [], [], []
                for group_name, population in zip(['population', 'survivors', 'failed'], [history, survivors,
                                                                                          failed]):
                    group = f[str(gen_index)][group_name]
                    for i in range(len(group)):
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
        print('PopulationStorage: loaded %i generations from file: %s' % (len(self.history), file_path))


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
            x0 = [None for i in range(len(bounds))]
        for i in range(len(x0)):
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
        for i in range(len(xmin)):
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
        for i in range(len(x)):
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
            print('Before: xi: %.4f, step: %.4f, xi_min: %.4f, xi_max: %.4f' % (xi, step, xi_min, xi_max))
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
            print('After: xi: %.4f, step: %.4f, xi_min: %.4f, xi_max: %.4f' % (new_xi, step, xi_min, xi_max))
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
            print('Before: log_xi: %.4f, step: %.4f, xi_logmin: %.4f, xi_logmax: %.4f' % (xi_log, step, xi_logmin,
                                                                                          xi_logmax))
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
            print('After: xi: %.4f, step: %.4f, xi_logmin: %.4f, xi_logmax: %.4f' % (new_xi, step, xi_logmin,
                                                                                     xi_logmax))
        return new_xi

    def apply_rel_bounds(self, x, stepsize, rel_bounds=None, disp=False):
        """

        :param x: array
        :param stepsize: float
        :param rel_bounds: list
        :param disp: bool
        """
        if disp:
            print('orig x: %s' % str(x))
        new_x = np.array(x)
        new_min = deepcopy(self.xmin)
        new_max = deepcopy(self.xmax)
        if rel_bounds is not None:
            for i, rel_bound_rule in enumerate(rel_bounds):
                dep_param = rel_bound_rule[0]  # Dependent param: name of the parameter that may be modified
                dep_param_ind = self.param_indexes[dep_param]
                if dep_param_ind >= len(x):
                    raise Exception('Dependent parameter index is out of bounds for rule %d.' % i)
                factor = rel_bound_rule[2]
                ind_param = rel_bound_rule[3]  # Independent param: name of the parameter that sets the bounds
                ind_param_ind = self.param_indexes[ind_param]
                if ind_param_ind >= len(x):
                    raise Exception('Independent parameter index is out of bounds for rule %d.' % i)
                if rel_bound_rule[1] == "=":
                    new_xi = factor * new_x[ind_param_ind]
                    if (new_xi >= self.xmin[dep_param_ind]) and (new_xi < self.xmax[dep_param_ind]):
                        new_x[dep_param_ind] = new_xi
                    else:
                        raise Exception('Relative bounds rule %d contradicts fixed parameter bounds.' % i)
                    continue
                if disp:
                    print('Before rel bound rule %i. xi: %.4f, min: %.4f, max: %.4f' % (i, new_x[dep_param_ind],
                                                                                        new_min[dep_param_ind],
                                                                                        new_max[dep_param_ind]))

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
                        print('After rel bound rule %i. xi: %.4f, min: %.4f, max: %.4f' % (i, new_xi,
                                                                                           new_min[dep_param_ind],
                                                                                           new_max[dep_param_ind]))
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
                    print('Parameter %d: value %.3f did not meet relative bound in rule %d.' % \
                          (dep_param_ind, x[dep_param_ind], r))
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
                 max_fitness=5, disp=False, hot_start=False, storage_file_path=None, **kwargs):
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
        elif isinstance(evaluate, basestring) and evaluate in globals() and isinstance(globals()[evaluate], collections.Callable):
            self.evaluate = globals()[evaluate]
        else:
            raise TypeError("PopulationAnnealing: evaluate must be callable.")
        if select is None:
            # self.select = select_survivors_by_rank_and_fitness  # select_survivors_by_rank
            self.select = select_survivors_population_annealing
        elif isinstance(select, collections.Callable):
            self.select = select
        elif isinstance(select, basestring) and select in globals() and isinstance(globals()[select], collections.Callable):
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
        elif isinstance(take_step, basestring) and take_step in globals() and \
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
                                (self.num_gen - 1))
            elif self.num_gen % self.path_length == 0:
                self.step_survivors()
            else:
                self.step_population()
            self.objectives_stored = False
            if self.disp:
                print('PopulationAnnealing: Gen %i, yielding parameters for population size %i' % \
                      (self.num_gen, len(self.population)))
            self.local_time = time.time()
            self.num_gen += 1
            sys.stdout.flush()
            yield [individual.x for individual in self.population]
        if not self.objectives_stored:
            raise Exception('PopulationAnnealing: objectives from final Gen %i were not stored or evaluated' %
                            (self.num_gen - 1))
        if self.disp:
            print('PopulationAnnealing: %i generations took %.2f s' % (self.max_gens, time.time() - self.start_time))
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
            print('PopulationAnnealing: Gen %i, computing features for population size %i took %.2f s; %i individuals' \
                  ' failed' % (self.num_gen - 1, len(self.population), time.time() - self.local_time, num_failed))
        self.local_time = time.time()
        self.population = filtered_population
        self.storage.append(self.population, survivors=self.survivors, failed=self.failed,
                            step_size=self.take_step.stepsize)
        self.objectives_stored = True
        if self.num_gen % self.path_length == 0:
            if len(self.population) > 0:
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
            print('PopulationAnnealing: Gen %i, evaluating iteration took %.2f s' % (self.num_gen - 1,
                                                                                     time.time() - self.local_time))
        self.local_time = time.time()

    def init_population(self):
        """

        """
        pop_size = self.pop_size
        if self.x0 is not None:
            self.population = []
            if self.num_gen == 0:
                self.population.append(Individual(self.x0))
                pop_size -= 1
            self.population.extend([Individual(self.take_step(self.x0, stepsize=1., wrap=True))
                                    for i in range(pop_size)])
        else:
            self.population = [Individual(x) for x in self.random.uniform(self.xmin, self.xmax, pop_size)]

    def step_survivors(self):
        """
        Consider the highest ranked Individuals of the previous iteration be survivors. Seed the next generation with
        steps taken from the set of survivors.
        """
        new_step_size = self.take_step.stepsize * self.adaptive_step_factor
        if self.disp:
            print('PopulationAnnealing: Gen %i, previous step_size: %.3f, new step_size: %.3f' % \
                  (self.num_gen, self.take_step.stepsize, new_step_size))
        self.take_step.stepsize = new_step_size
        new_population = []
        if not self.survivors:
            self.init_population()
        else:
            # num_survivors = min(self.num_survivors, len(self.survivors))
            num_survivors = len(self.survivors)
            for i in range(self.pop_size):
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
            for i in range(self.pop_size):
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
        print('%s:' % name)
        print('params:')
        pprint.pprint(self.x_dict[name])
        print('features:')
        pprint.pprint(self.features[name])
        print('objectives:')
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
    if pop_size == 0:
        return min_objectives, max_objectives
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
    for m in range(num_objectives):
        indexes = list(range(pop_size))
        objective_vals = [individual.objectives[m] for individual in population]
        indexes.sort(key=objective_vals.__getitem__)
        new_population = list(map(population.__getitem__, indexes))

        # keep the borders
        new_population[0].distance += 1.e15
        new_population[-1].distance += 1.e15

        objective_min = new_population[0].objectives[m]
        objective_max = new_population[-1].objectives[m]

        if objective_min != objective_max:
            for i in range(1, pop_size - 1):
                new_population[i].distance += (new_population[i + 1].objectives[m] -
                                               new_population[i - 1].objectives[m]) / \
                                              (objective_max - objective_min)


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
    indexes = list(range(pop_size))
    distances = [individual.distance for individual in population]
    indexes.sort(key=distances.__getitem__)
    indexes.reverse()
    population = list(map(population.__getitem__, indexes))
    return population


def assign_absolute_energy(population):
    """
    Modifies in place the energy attribute of each Individual in the population. Energy is assigned as the sum across
    all non-normalized objectives.
    :param population: list of :class:'Individual'
    """
    indexes = list(range(len(population)))
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
    indexes = list(range(pop_size))
    indexes.sort(key=energy_vals.__getitem__)
    population = list(map(population.__getitem__, indexes))
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
    for m in range(num_objectives):
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
    for fitness in range(max_fitness + 1):
        new_front = [individual for individual in population if individual.fitness == fitness]
        assign_relative_energy(new_front)


def assign_rank_by_fitness_and_energy(population):
    """
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
    for fitness in range(max_fitness + 1):
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
    indexes = list(range(pop_size))
    indexes.sort(key=rank_vals.__getitem__)
    new_population = list(map(population.__getitem__, indexes))
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
        for fitness in range(max_fitness + 1):
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

        for p in range(len(population)):
            S[p] = []  # list of Individuals that p dominates
            n[p] = 0  # number of Individuals that dominate p

            for q in range(len(population)):
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
            F[i + 1] = []  # next front
            # take the elements from the previous front
            for p in F[i]:
                # take the elements that p dominates
                for q in S[p]:
                    # decrease domination value of all Individuals that p dominates
                    n[q] -= 1
                    if n[q] == 0:
                        population[q].fitness = i + 1  # assign fitness of current front
                        F[i + 1].append(q)
            i += 1
    else:
        for individual in population:
            individual.fitness = 0
    if disp:
        print(F)


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
    rank_vals = list(range(len(population)))
    np.random.shuffle(rank_vals)
    for i, individual in enumerate(population):
        rank = rank_vals[i]
        individual.rank = rank
        if disp:
            print('Individual %i: rank %i, x: %s' % (i, rank, individual.x))


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
    for fitness in range(max_fitness + 1):
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
            raise Exception('select_survivors_population_annealing: objectives have not been stored for all '
                            'Individuals in population')
        num_objectives = max(num_objectives)

        specialists = []
        for m in range(num_objectives):
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


def init_controller_context(config_file_path=None, storage_file_path=None, export_file_path=None, param_gen=None,
                            label=None, analyze=None, output_dir=None, **kwargs):
    """

    :param config_file_path: str (path)
    :param storage_file_path: str (path)
    :param export_file_path: str (path)
    :param param_gen: str
    :param label: str
    :param analyze: bool
    :param output_dir: str (dir)
    """
    context = find_context()
    if config_file_path is not None:
        context.config_file_path = config_file_path
    if 'config_file_path' not in context() or context.config_file_path is None or \
            not os.path.isfile(context.config_file_path):
        raise Exception('nested.optimize: config_file_path specifying required optimization parameters is missing or '
                        'invalid.')
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
        context.label = ''
    else:
        context.label = '_' + context.label
    if param_gen is not None:
        context.param_gen = param_gen
    context.ParamGenClassName = context.param_gen
    # ParamGenClass points to the parameter generator class, while ParamGenClassName points to its name as a string
    if context.ParamGenClassName not in globals():
        raise Exception('nested.optimize: %s has not been imported, or is not a valid class of parameter '
                        'generator.' % context.ParamGenClassName)
    context.ParamGenClass = globals()[context.ParamGenClassName]
    if output_dir is not None:
        context.output_dir = output_dir
    if 'output_dir' not in context():
        context.output_dir = None
    if context.output_dir is None:
        output_dir_str = ''
    else:
        output_dir_str = context.output_dir + '/'
    if storage_file_path is not None:
        context.storage_file_path = storage_file_path
    if 'storage_file_path' not in context() or context.storage_file_path is None:
        context.storage_file_path = '%s%s_%s%s_%s_optimization_history.hdf5' % \
                                    (output_dir_str, datetime.datetime.today().strftime('%Y%m%d_%H%M'),
                                     context.optimization_title, context.label, context.ParamGenClassName)
    if export_file_path is not None:
        context.export_file_path = export_file_path
    if 'export_file_path' not in context() or context.export_file_path is None:
        context.export_file_path = '%s%s_%s%s_%s_optimization_exported_output.hdf5' % \
                                   (output_dir_str, datetime.datetime.today().strftime('%Y%m%d_%H%M'),
                                    context.optimization_title, context.label, context.ParamGenClassName)

    context.sources = set([elem[0] for elem in context.update_context_list] + list(context.get_objectives_dict.keys()) +
                          [stage['source'] for stage in context.stages if 'source' in stage])
    context.reset_worker_funcs = []
    context.shutdown_worker_funcs = []
    for source in context.sources:
        m = importlib.import_module(source)
        m_context_name = find_context_name(source)
        setattr(m, m_context_name, context)
        if hasattr(m, 'reset_worker'):
            reset_func = getattr(m, 'reset_worker')
            if not isinstance(reset_func, collections.Callable):
                raise Exception('nested.optimize: reset_worker for source: %s is not a callable function.' % source)
            context.reset_worker_funcs.append(reset_func)
        if hasattr(m, 'shutdown_worker'):
            shutdown_func = getattr(m, 'shutdown_worker')
            if not isinstance(shutdown_func, collections.Callable):
                raise Exception('nested.optimize: shutdown_worker for source: %s is not a callable function.' % source)
            context.shutdown_worker_funcs.append(shutdown_func)

    context.update_context_funcs = []
    for source, func_name in context.update_context_list:
        module = sys.modules[source]
        func = getattr(module, func_name)
        if not isinstance(func, collections.Callable):
            raise Exception('nested.optimize: update_context: %s for source: %s is not a callable function.'
                            % (func_name, source))
        context.update_context_funcs.append(func)
    context.group_sizes = []
    for stage in context.stages:
        source = stage['source']
        module = sys.modules[source]
        if 'group_size' in stage and stage['group_size'] is not None:
            context.group_sizes.append(stage['group_size'])
        else:
            context.group_sizes.append(1)
        if 'get_args_static' in stage and stage['get_args_static'] is not None:
            func_name = stage['get_args_static']
            func = getattr(module, func_name)
            if not isinstance(func, collections.Callable):
                raise Exception('nested.optimize: get_args_static: %s for source: %s is not a callable function.'
                                % (func_name, source))
            stage['get_args_static_func'] = func
        elif 'get_args_dynamic' in stage and stage['get_args_dynamic'] is not None:
            func_name = stage['get_args_dynamic']
            func = getattr(module, func_name)
            if not isinstance(func, collections.Callable):
                raise Exception('nested.optimize: get_args_dynamic: %s for source: %s is not a callable function.'
                                % (func_name, source))
            stage['get_args_dynamic_func'] = func
        if 'compute_features' in stage and stage['compute_features'] is not None:
            func_name = stage['compute_features']
            func = getattr(module, func_name)
            if not isinstance(func, collections.Callable):
                raise Exception('nested.optimize: compute_features: %s for source: %s is not a callable function.'
                                % (func_name, source))
            stage['compute_features_func'] = func
        elif 'compute_features_shared' in stage and stage['compute_features_shared'] is not None:
            func_name = stage['compute_features_shared']
            func = getattr(module, func_name)
            if not isinstance(func, collections.Callable):
                raise Exception('nested.optimize: compute_features_shared: %s for source: %s is not a callable '
                                'function.' % (func_name, source))
            stage['compute_features_shared_func'] = func
        if 'filter_features' in stage and stage['filter_features'] is not None:
            func_name = stage['filter_features']
            func = getattr(module, func_name)
            if not isinstance(func, collections.Callable):
                raise Exception('nested.optimize: filter_features: %s for source: %s is not a callable function.'
                                % (func_name, source))
            stage['filter_features_func'] = func
        if 'synchronize' in stage and stage['synchronize'] is not None:
            func_name = stage['synchronize']
            func = getattr(module, func_name)
            if not isinstance(func, collections.Callable):
                raise Exception('nested.optimize: synchronize: %s for source: %s is not a callable function.'
                                % (func_name, source))
            stage['synchronize_func'] = func
    context.get_objectives_funcs = []
    for source, func_name in viewitems(context.get_objectives_dict):
        module = sys.modules[source]
        func = getattr(module, func_name)
        if not isinstance(func, collections.Callable):
            raise Exception('nested.optimize: get_objectives: %s for source: %s is not a callable function.'
                            % (func_name, source))
        context.get_objectives_funcs.append(func)
    if analyze is not None:
        context.analyze = analyze
    if 'analyze' in context() and context.analyze:
        context.pop_size = 1


def init_worker_contexts(sources, update_context_funcs, param_names, default_params, feature_names, objective_names,
                         target_val, target_range, export_file_path, output_dir, disp, optimization_title=None,
                         label=None, **kwargs):
    """

    :param sources: set of str (source names)
    :param update_context_funcs: list of callable
    :param param_names: list of str
    :param default_params: dict
    :param feature_names: list of str
    :param objective_names: list of str
    :param target_val: dict
    :param target_range: dict
    :param export_file_path: str (path)
    :param output_dir: str (dir path)
    :param disp: bool
    :param optimization_title: str
    :param label: str
    """
    context = find_context()
    if output_dir is not None:
        context.output_dir = output_dir
    if 'output_dir' not in context():
        context.output_dir = None
    if context.output_dir is None:
        output_dir_str = ''
    else:
        output_dir_str = context.output_dir + '/'
    temp_output_path = '%snested_optimize_temp_output_%s_pid%i.hdf5' % \
                       (output_dir_str, datetime.datetime.today().strftime('%Y%m%d_%H%M'), os.getpid())
    context.update(locals())
    context.update(kwargs)
    if 'interface' in context():
        if hasattr(context.interface, 'comm'):
            context.comm = context.interface.comm
        if hasattr(context.interface, 'worker_comm'):
            context.worker_comm = context.interface.worker_comm
        if hasattr(context.interface, 'global_comm'):
            context.global_comm = context.interface.global_comm
        if hasattr(context.interface, 'num_workers'):
            context.num_workers = context.interface.num_workers
    elif 'comm' not in context():
        try:
            from mpi4py import MPI
            context.comm = MPI.COMM_WORLD
        except Exception:
            pass
    for source in sources:
        m = importlib.import_module(source)
        m_context_name = find_context_name(source)
        setattr(m, m_context_name, context)
        if hasattr(m, 'config_worker'):
            config_func = getattr(m, 'config_worker')
            if not isinstance(config_func, collections.Callable):
                raise Exception('nested.optimize: init_worker_contexts: source: %s; problem executing config_worker' %
                                source)
            config_func()
    sys.stdout.flush()


def config_optimize_interactive(source_file_name, config_file_path=None, output_dir=None, temp_output_path=None,
                                export=False, export_file_path=None, label=None, disp=True, is_controller=False,
                                **kwargs):
    """
    nested.optimize is meant to be executed as a module, and refers to a config_file to import required submodules and
    create a workflow for optimization. During development of submodules, it is useful to be able to execute a submodule
    as a standalone script (as '__main__'). config_optimize_interactive allows a single process to properly parse the
    config_file and initialize a Context for testing purposes.
    # :param context: :class:'Context'
    :param source_file_name: str (filename of calling module)
    :param config_file_path: str (.yaml file path)
    :param output_dir: str (dir path)
    :param temp_output_path: str (.hdf5 file path)
    :param export: bool
    :param export_file_path: str (.hdf5 file path)
    :param label: str
    :param disp: bool
    :param is_controller: bool
    """
    context = find_context()
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
                                   (output_dir_str, datetime.datetime.today().strftime('%Y%m%d_%H%M'), os.getpid(),
                                    context.optimization_title, label)
    context.export = export
    if export_file_path is not None:
        context.export_file_path = export_file_path
    if 'export_file_path' not in context() or context.export_file_path is None:
        context.export_file_path = '%s%s_%s%s_interactive_exported_output.hdf5' % \
                                   (output_dir_str, datetime.datetime.today().strftime('%Y%m%d_%H%M'),
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
    context.sources = [local_source]

    if 'interface' in context():
        if hasattr(context.interface, 'comm'):
            context.comm = context.interface.comm
        if hasattr(context.interface, 'worker_comm'):
            context.worker_comm = context.interface.worker_comm
        if hasattr(context.interface, 'global_comm'):
            context.global_comm = context.interface.global_comm
        if hasattr(context.interface, 'num_workers'):
            context.num_workers = context.interface.num_workers
    elif 'comm' not in context():
        try:
            from mpi4py import MPI
            context.comm = MPI.COMM_WORLD
        except Exception:
            print('ImportWarning: nested.optimize: source: %s; config_optimize_interactive: problem importing from ' \
                  'mpi4py' % local_source)
    if 'num_workers' not in context():
        context.num_workers = 1
    if not is_controller:
        if hasattr(m, 'config_worker'):
            config_func = getattr(m, 'config_worker')
            if not isinstance(config_func, collections.Callable):
                raise Exception('nested.optimize: source: %s; config_optimize_interactive: problem executing '
                                'config_worker' % local_source)
            config_func()
        update_source_contexts(context.x0_array, context)


def config_parallel_interface(source_file_name, config_file_path=None, output_dir=None, temp_output_path=None,
                              export=False, export_file_path=None, label=None, disp=True, **kwargs):
    """
    nested.parallel is used for parallel map operations. This method imports optional parameters from a config_file and
    initializes a Context object on each worker.
    :param source_file_name: str (filename of calling module)
    :param config_file_path: str (.yaml file path)
    :param output_dir: str (dir path)
    :param temp_output_path: str (.hdf5 file path)
    :param export: bool
    :param export_file_path: str (.hdf5 file path)
    :param label: str
    :param disp: bool
    """
    context = find_context()
    if config_file_path is not None:
        context.config_file_path = config_file_path
    if 'config_file_path' in context() and context.config_file_path is not None:
        if not os.path.isfile(context.config_file_path):
            raise Exception('nested.parallel: config_file_path specifying optional is invalid.')
        else:
            config_dict = read_from_yaml(context.config_file_path)
    else:
        config_dict = {}
    context.update(config_dict)
    context.kwargs = config_dict  # Extra arguments to be passed to imported sources

    if label is not None:
        context.label = label
    if 'label' not in context() or context.label is None:
        context.label = ''
    else:
        context.label = '_' + context.label

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
        context.temp_output_path = '%s%s_pid%i%s_temp_output.hdf5' % \
                                   (output_dir_str, datetime.datetime.today().strftime('%Y%m%d_%H%M'), os.getpid(),
                                    context.label)
    context.export = export
    if export_file_path is not None:
        context.export_file_path = export_file_path
    if 'export_file_path' not in context() or context.export_file_path is None:
        context.export_file_path = '%s%s%s_exported_output.hdf5' % \
                                   (output_dir_str, datetime.datetime.today().strftime('%Y%m%d_%H%M'), context.label)
    context.disp = disp

    local_source = os.path.basename(source_file_name).split('.')[0]
    m = sys.modules['__main__']
    context.sources = [local_source]

    if 'comm' not in context():
        try:
            from mpi4py import MPI
            context.comm = MPI.COMM_WORLD
        except Exception:
            print('ImportWarning: nested.parallel: source: %s; config_parallel_interface: problem importing from ' \
                  'mpi4py' % local_source)

    if hasattr(m, 'config_worker'):
        config_func = getattr(m, 'config_worker')
        if not isinstance(config_func, collections.Callable):
            raise Exception('nested.parallel: source: %s; config_parallel_interface: problem executing config_worker' %
                            local_source)
        config_func()


def merge_exported_data_from_yaml(yaml_file_path, new_file_name=None, data_dir=None, verbose=True):
    """
    Load a list of .hdf5 file names from a .yaml file and merge into a single .hdf5 file.
    :param yaml_file_path: str (path)
    :param new_file_name: str (path)
    :param data_dir: str (path)
    :param verbose: bool
    """
    if not os.path.isfile(yaml_file_path):
        raise Exception('merge_exported_data_from_yaml: missing yaml_file at specified path: %s' % yaml_file_path)
    file_path_list = read_from_yaml(yaml_file_path)
    if not len(file_path_list) > 0:
        if verbose:
            print('merge_exported_data: no data exported; empty file_path_list')
        return None
    if new_file_name is None:
        new_file_path = 'merged_exported_data_%s_%i.hdf5' % \
                        (datetime.datetime.today().strftime('%m%d%Y%H%M'), os.getpid())
    else:
        new_file_path = new_file_name
    if data_dir is not None:
        if not os.path.isdir(data_dir):
            raise Exception('merge_exported_data_from_yaml: cannot find data_dir: %s' % data_dir)
        file_path_list = ['%s/%s' % (data_dir, file_name) for file_name in file_path_list]
        new_file_path = '%s/%s' % (data_dir, new_file_path)
    return merge_exported_data(file_path_list, new_file_path, verbose=verbose)


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
        new_file_path = 'merged_exported_data_%s_%i.hdf5' % \
                        (datetime.datetime.today().strftime('%m%d%Y%H%M'), os.getpid())
    if not len(file_path_list) > 0:
        if verbose:
            print('merge_exported_data: no data exported; empty file_path_list')
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
                                print('enumerated', group, old_f[group], target)
                            for source in viewvalues(old_f[group]):
                                target.copy(source, target, name=str(enum))
                                enum += 1
                        else:
                            if verbose:
                                print('not enumerated', group, old_f[group], target)
                            h5_nested_copy(old_f[group], target)
    if verbose:
        print('merge_exported_data: exported to file_path: %s' % new_file_path)
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
        for key, val in viewitems(source):
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
        local_context = find_context()
    if hasattr(local_context, 'update_context_funcs'):
        local_context.x_array = x
        for update_func in local_context.update_context_funcs:
            update_func(x, local_context)


"""
--------------------------functions to plot local sensitivity-------------------
to call the function:
    from nested.optimize_utils import * 
    pop = PopulationStorage(file_path='path_to_hdf5_file.hdf5')
    # returns PopulationStorage object with perturbations and LSA object to interrogate
    perturbation_vectors, LSA = local_sensitivity(pop) 
"""


def pop_to_matrix(population, feat_bool):
    """converts collection of individuals in PopulationStorage into a matrix for data manipulation

    :param population: PopulationStorage object
    :param feat_bool: True if we're doing LSA on features, False if on objectives
    :return: data: 2d array. rows = each data point or individual, col = parameters, then features
    """
    data = []
    generation_array = population.history
    for generation in generation_array:
        for datum in generation:
            x_array = datum.x
            y_array = datum.features if feat_bool else datum.objectives
            individual_array = np.append(x_array, y_array, axis=0)

            data.append(individual_array)
    return np.array(data)


def process_data(data):
    """need to log normalize parts of the data, so processing columns that are negative and/or have zeros is needed"""
    processed_data = np.copy(data)
    neg = list(set(np.where(data < 0)[1]))
    pos = list(set(np.where(data > 0)[1]))
    z = list(set(np.where(data == 0)[1]))
    crossing = [num for num in pos if num in neg]
    pure_neg = [num for num in neg if num not in pos]

    # transform data
    processed_data[:, pure_neg] *= -1

    # diff = np.max(data, axis=0) - np.min(data, axis=0)
    # diff[np.where(diff == 0)[0]] = 1.
    # magnitude = np.log10(diff)
    # offset = 10 ** (magnitude - 2)
    # processed_data[:, z] += offset[z]

    return processed_data, crossing, z


def order_the_dict(x0_dict, names):
    """
    HallOfFame = dict with dicts, therefore ordering is different from .yaml file.
    x0_dict = dict from HallOfFame: key = string (name), val = real number
    name = list of input variable names from .yaml file
    this orders the values in the way that the .yaml file is
    """
    ordered_list = [None] * len(names)
    for k, v in viewitems(x0_dict):
        index = names.index(k)
        ordered_list[index] = v
    return np.asarray(ordered_list)


def x0_to_array(population, x0_string, param_names, data, processed_data):
    """
    from x0 string (e.g. 'best'), returns the respective array/data which contains
    both the parameter and output values
    """
    fame = HallOfFame(population)
    num_param = len(fame.param_names)

    x0_x_dict = fame.x_dict.get(x0_string)
    x0_x_array = order_the_dict(x0_x_dict, param_names.tolist())

    index = np.where(data[:, :num_param] == x0_x_array)[0][0]
    return processed_data[index, :], num_param


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

    return logmin_array, logdiff_array, logmax_array


def normalize_data(population, data, processed_data, crossing, z, x0_string, param_names, input_is_not_param,
                   norm_search=False):
    """normalize all data points. used for calculating neighborship

    :param population: PopulationStorage object
    :param data: 2d array object with data from generations
    :param processed_data: data has been transformed for the cols that need to be log-normalized such that the values
                           can be logged
    :param crossing: list of column indices such that within the column, values cross 0
    :param z: list of column idx such that column has a 0
    :param x0_string: user input string specifying x0
    :param param_names: names of parameters
    :param input_is_not_param: bool
    :param linspace_search: bool, whether neighbor search should be done in a linear space
    :return: matrix of normalized values for parameters and features
    """
    # process_data DOES NOT process the columns (ie, parameters and features) that cross 0, because
    # that col will just be lin normed.
    warnings.simplefilter("ignore")

    x0_array, num_param = x0_to_array(population, x0_string, param_names, data, processed_data)
    x0_normed = np.copy(x0_array)
    x0_log = np.log10(np.copy(x0_array))

    data_normed = np.copy(processed_data)
    num_rows, num_cols = processed_data.shape

    min_array, diff_array = get_linear_arrays(processed_data)
    diff_array[np.where(diff_array == 0)[0]] = 1
    data_log_10 = np.log10(np.copy(processed_data))
    logmin_array, logdiff_array, logmax_array = get_log_arrays(data_log_10)

    # move out
    scaling = None  # holds a list of whether the column was log or lin normalized (string)
    if norm_search:
        scaling = np.array(['log'] * num_cols)
        scaling[np.where(logdiff_array < 2)[0]] = 'lin'
        scaling[crossing] = 'lin';
        scaling[z] = 'lin'
        lin_loc = np.where(scaling == 'lin')[0]
        log_loc = np.where(scaling == 'log')[0]

        data_normed[:, lin_loc] = np.true_divide((processed_data[:, lin_loc] - min_array[lin_loc]), diff_array[lin_loc])
        x0_normed[lin_loc] = np.true_divide((x0_normed[lin_loc] - min_array[lin_loc]), diff_array[lin_loc])
        data_normed[:, log_loc] = np.true_divide((data_log_10[:, log_loc] - logmin_array[log_loc]),
                                                 logdiff_array[log_loc])
        x0_normed[log_loc] = np.true_divide((x0_log[log_loc] - logmin_array[log_loc]), logdiff_array[log_loc])

        data_normed = np.nan_to_num(data_normed)

    best_normed = np.array(np.nan_to_num(x0_normed))
    X_x0 = x0_array[num_param:] if input_is_not_param else x0_array[:num_param]
    packaged_variables = [X_x0, scaling, logdiff_array, logmin_array, diff_array, min_array]
    print("Data normalized")
    return data_normed, best_normed, packaged_variables


def get_important_inputs(data, num_input, num_output, num_param, input_names, y_names, input_is_not_param,
                         inp_out_same):
    """using decision trees, get important parameters for each output.
    "feature," in this case, is used in the same way one would use "parameter"

    :param data: 2d array, un-normalized
    :param num_input: int
    :param num_output: int, number of features or objectives
    :param num_param: int
    :param input_names: list of strings
    :param y_names: list of strings representing names of features or objectives
    :param input_is_not_param: bool
    :param inp_out_same: bool
    :return: important parameters - a list of lists. list length = num_features
    """
    # the sum of feature_importances_ is 1, so the baseline should be relative to num_input
    # the below calculation is pretty ad hoc and based fitting on (20, .1), (200, .05), (2000, .01); (num_input, baseline)
    baseline = 0.15688 - 0.0195433 * np.log(num_input)
    if baseline < 0: baseline = .005

    y = data[:, num_param:]
    X = data[:, num_param:] if input_is_not_param else data[:, :num_param]
    important_inputs = [[] for _ in range(num_output)]

    # create a decision tree for each feature. each independent var is considered "important" if over the baseline
    for i in range(num_output):
        dt = DecisionTreeRegressor(random_state=0, max_depth=200)
        Xi = X[:, [x for x in range(num_input) if x != i]] if inp_out_same else X
        dt.fit(Xi, y[:, i])

        input_list = list(zip([round(t, 4) for t in dt.feature_importances_], input_names))
        for j in range(len(dt.feature_importances_)):
            if dt.feature_importances_[j] > baseline:
                important_inputs[i].append(input_list[j][1])  # append the name of the param (str)
        if inp_out_same: important_inputs[i].append(input_names[i])

    print("Important dependent variables calculated:")
    for i in range(num_output):
        print(y_names[i], "-", important_inputs[i])
    return important_inputs


def check_dominant(feat_imp, imp_loc, unimp_loc):
    imp_mean = np.mean(feat_imp[imp_loc])
    unimp_mean = np.mean(feat_imp[unimp_loc])
    if imp_mean != 0 and unimp_mean != 0 and len(imp_loc) != 0 and int(math.log10(imp_mean)) - \
            int(math.log10(unimp_mean)) >= 2:
        return True
    return False


def get_important_inputs2(data, num_input, num_output, num_param, input_names, y_names, input_is_not_param,
                          inp_out_same, relaxed_factor):
    """using decision trees, get important parameters for each output.
    "feature," in this case, is used in the same way one would use "parameter"

    :param data: 2d array, un-normalized
    :param num_input: int
    :param num_output: int, number of features or objectives
    :param num_param: int
    :param input_names: list of strings
    :param y_names: list of strings representing names of features or objectives
    :param input_is_not_param: bool
    :param inp_out_same: bool
    :return: important parameters - a list of lists. list length = num_features
    """
    # the sum of feature_importances_ is 1, so the baseline should be relative to num_input
    # the below calculation is pretty ad hoc and based fitting on (20, .1), (200, .05), (2000, .01); (num_input, baseline)
    baseline = 0.15688 - 0.0195433 * np.log(num_input)
    if baseline < 0: baseline = .005

    y = data[:, num_param:]
    X = data[:, num_param:] if input_is_not_param else data[:, :num_param]
    important_inputs = [[] for _ in range(num_output)]
    unimp_inputs = [[] for _ in range(num_output)]
    dominant_list = [1.] * num_input

    # create a decision tree for each feature. each independent var is considered "important" if over the baseline
    for i in range(num_output):
        dt = DecisionTreeRegressor(random_state=0, max_depth=200)
        Xi = X[:, [x for x in range(num_input) if x != i]] if inp_out_same else X
        dt.fit(Xi, y[:, i])

        # input_list = np.array(list(zip(map(lambda t: round(t, 4), dt.feature_importances_), input_names)))
        imp_loc = np.where(dt.feature_importances_ >= baseline)[0]
        unimp_loc = np.where(dt.feature_importances_ < baseline)[0]
        important_inputs[i] = input_names[imp_loc].tolist()
        unimp_inputs[i] = input_names[unimp_loc]

        if inp_out_same:
            important_inputs[i].append(input_names[i])
            imp_loc[np.where(imp_loc > i)[0]] = imp_loc[np.where(imp_loc > i)[0]] - 1    #shift for check_dominant
            unimp_loc[np.where(imp_loc > i)[0]] = unimp_loc[np.where(imp_loc > i)[0]] - 1
        if check_dominant(dt.feature_importances_, imp_loc, unimp_loc): dominant_list[i] = relaxed_factor

    print("Important dependent variables calculated:")
    for i in range(num_output):
        print(y_names[i], "-", important_inputs[i])
    return important_inputs, dominant_list


def split_parameters(num_input, important_inputs_set, input_names, p):
    # convert str to int (idx)
    if len(important_inputs_set) > 0:
        input_indices = [np.where(input_names == inp)[0][0] for inp in important_inputs_set]
    else:  # no important parameters
        return [], [x for x in range(num_input)]

    # create subsets of the input matrix based on importance. leave out query var from the sets
    important = [x for x in input_indices if x != p]
    unimportant = [x for x in range(num_input) if x not in important and x != p]
    return important, unimportant


def possible_neighbors(important, unimportant, X_normed, X_x0_normed, important_rad, unimportant_rad):
    """make two BallTrees to do distance querying"""
    # get first set of neighbors (filter by important params)
    # second element of the tree query is dtype, which is useless
    if important:
        important_cheb_tree = BallTree(X_normed[:, important], metric='chebyshev')
        important_neighbor_array = important_cheb_tree.query_radius(X_x0_normed[important].reshape(1, -1),
                                                                    r=important_rad)[0]
    else:
        important_neighbor_array = np.array([])

    # get second set (by unimprt parameters)
    if unimportant:
        unimportant_tree = BallTree(X_normed[:, unimportant], metric='euclidean')
        unimportant_neighbor_array = unimportant_tree.query_radius(X_x0_normed[unimportant].reshape(1, -1),
                                                                   r=unimportant_rad)[0]
    else:
        unimportant_neighbor_array = np.array([])

    return unimportant_neighbor_array, important_neighbor_array


def update_debugger(debug_matrix, unimportant_neighbor_array, important_neighbor_array, filtered_neighbors,
                    passed_neighbors, i, o):
    debug_matrix[i][o]['SIG'] = filtered_neighbors
    debug_matrix[i][o]['ALL'] = passed_neighbors

    unimp_set = set(unimportant_neighbor_array)
    imp_set = set(important_neighbor_array)
    debug_matrix[i][o]['UI'] = unimp_set - imp_set
    debug_matrix[i][o]['I'] = imp_set - unimp_set

    # get overlap
    # ncols = unimportant_neighbor_array.shape[1] if len(unimportant_neighbor_array.shape) > 1 else unimportant_neighbor_array.shape[0]
    # dtype = {'names': ['f{}'.format(i) for i in range(ncols)], 'formats': ncols * [unimportant_neighbor_array.dtype]}
    # tmp = np.intersect1d(unimportant_neighbor_array.view(dtype), important_neighbor_array.view(dtype))
    # debug_matrix[i][o]['DIST'] = tmp.view(unimportant_neighbor_array.dtype).reshape(-1, ncols)
    debug_matrix[i][o]['DIST'] = unimp_set & imp_set

    return debug_matrix


def filter_neighbors(x_not, important_neighbor_array, unimportant_neighbor_array, X_normed, X_x0_normed,
                     important_rad, i, o, debug_matrix):
    """filter according to the radii constraints and if query parameter perturbation > twice the max perturbation
    of important parameters
    passed neighbors = passes all constraints
    filtered neighbors = neighbors that fit the important input variable distance constraint + the distance of
        the input variable of interest is more than twice that of the important variable constraint"""
    #if not len(unimportant_neighbor_array): return [x_not], debug_matrix
    #if not len(important_neighbor_array): return [x_not], debug_matrix

    if len(unimportant_neighbor_array) > 1 and len(important_neighbor_array) > 1:
        sig_perturbation = abs(X_normed[important_neighbor_array, i] - X_x0_normed[i]) >= 2 * important_rad
        filtered_neighbors = important_neighbor_array[sig_perturbation].tolist() + [x_not]
        passed_neighbors = [idx for idx in filtered_neighbors if idx in unimportant_neighbor_array]
    else:
        filtered_neighbors = [x_not]
        passed_neighbors = [x_not]

    debug_matrix = update_debugger(debug_matrix, unimportant_neighbor_array, important_neighbor_array,
                                   filtered_neighbors, passed_neighbors, i, o)

    return passed_neighbors, debug_matrix


def check_range(input_indices, input_range, filtered_neighbors, X_x0_normed, X_normed):
    subset_X = X_normed[list(filtered_neighbors), :]
    subset_X = subset_X[:, list(input_indices)]

    max_elem = np.max(np.abs(subset_X - X_x0_normed[input_indices]))
    min_elem = np.min(np.abs(subset_X - X_x0_normed[input_indices]))

    return min(min_elem, input_range[0]), max(max_elem, input_range[1])


def print_search_output(verbose, input, output, important_rad, filtered_neighbors, unimportant_rad):
    if verbose:
        print("\nInput:", input, "/ Output:", output)
        print("Max distance (for important parameters):", important_rad)
        print("Neighbors:", len(filtered_neighbors))
        print("Euclidean distance for unimportant parameters:", unimportant_rad)


def check_confounding(filtered_neighbors, X_x0_normed, X_normed, input_names, p):
    """
    a param is considered a possible confound if its count is greater than that of the query param

    sets up the second heatmap in the plot function, so it looks at three things: 1) confound 2) confound, but the
    parameter in the parameter/output pair was considered important to the output by DT, and 3) no neighbors found
    for param/output pair
    """
    # create dict with k=input, v=count of times that input var was the max perturbation in a point in the neighborhood
    max_inp_indices = {}
    for index in filtered_neighbors:
        diff = np.abs(X_x0_normed - X_normed[index])
        max_index = np.where(diff == np.max(diff))[0][0]
        if max_index in max_inp_indices:
            max_inp_indices[max_index] += 1
        else:
            max_inp_indices[max_index] = 1
    # print counts and keep a list of possible confounds to be checked later
    if p in max_inp_indices:
        query_param_count = max_inp_indices[p]
    else:
        query_param_count = 0
    possible_confound = []
    print("Count of greatest perturbation for each point in set of neighbors:")
    for k, v in viewitems(max_inp_indices):
        print(input_names[k], v)
        if v > query_param_count:
            possible_confound.append(k)
    return possible_confound


def get_neighbors(important, unimportant, X_normed, X_x0_normed, important_rad, unimportant_rad, x_not, i, o,
                  debugger_matrix):
    unimportant_neighbor_array, important_neighbor_array = possible_neighbors(
        important, unimportant, X_normed, X_x0_normed, important_rad, unimportant_rad)
    filtered_neighbors, debugger_matrix = filter_neighbors(
        x_not, important_neighbor_array, unimportant_neighbor_array, X_normed, X_x0_normed,
        important_rad, i, o, debugger_matrix)

    return filtered_neighbors, debugger_matrix


def housekeeping(neighbor_matrix, p, o, filtered_neighbors, verbose, input_names, y_names, important_rad,
                 unimportant_rad, important_range, unimportant_range, confound_matrix, X_x0_normed, X_normed,
                 important_indices, unimportant_indices):
    neighbor_matrix[p][o] = filtered_neighbors
    print_search_output(
        verbose, input_names[p], y_names[o], important_rad, filtered_neighbors, unimportant_rad)

    important_range = check_range(important_indices, important_range, filtered_neighbors, X_x0_normed, X_normed)
    unimportant_range = check_range(unimportant_indices, unimportant_range, filtered_neighbors, X_x0_normed, X_normed)
    confound_matrix[p][o] = check_confounding(
        filtered_neighbors, X_x0_normed, X_normed, input_names, p)

    return neighbor_matrix, important_range, unimportant_range, confound_matrix


def compute_neighbor_matrix(num_inputs, num_output, num_param, important_inputs, input_names, y_names, X_normed,
                            x0_normed, verbose, n_neighbors, max_dist, input_is_not_param, inp_out_same, dominant_list):
    """get neighbors for each feature/parameter pair based on 1) a max radius for important features and 2) a
    summed euclidean dist for unimportant parameters

    :param num_inputs: int
    :param num_output: int, num of features or objectives
    :param num_param: int
    :param important_inputs: list of lists of strings
    :param input_names: list of strings
    :param y_names: list of strings representing names of features or objectives
    :param X_normed: 2d array
    :param x0_normed: 1d array
    :param verbose: bool. print statements if true
    :param n_neighbors: int
    :param max_dist: starting point for important parameter radius
    :param inp_out_same: True if doing feature vs feature comparison
    :return: neighbor matrix, 2d array with each cell a list of integers (integers = neighbor indices in data matrix)
    :return:
    """
    IMP_RAD_CUTOFF = .3
    UNIMP_RAD_INCREMENT = .05
    UNIMP_RAD_START = .1
    UNIMP_UPPER_BOUND = [1., 1.3, 1.7, 2.3, 2.6]
    IMP_RAD_THRESHOLDS = [.08, .12]

    # initialize
    neighbor_matrix = np.empty((num_inputs, num_output), dtype=object)
    important_range = (float('inf'), float('-inf'))  # first element = min, second = max
    unimportant_range = (float('inf'), float('-inf'))
    confound_matrix = np.empty((num_inputs, num_output), dtype=object)
    debugger_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    #  constants
    X_x0_normed = x0_normed[num_param:] if input_is_not_param else x0_normed[:num_param]
    x_not = np.where(X_normed == X_x0_normed)[0][0]
    magnitude = int(math.log10(max_dist))

    for p in range(num_inputs):  # row
        for o in range(num_output):  # col
            if inp_out_same and p == o: continue
            important_rad = max_dist

            # split important vs unimportant parameters
            important, unimportant = split_parameters(num_inputs, important_inputs[o], input_names, p)
            filtered_neighbors = []
            while len(filtered_neighbors) < n_neighbors:
                unimportant_rad = UNIMP_RAD_START

                # break if most of the important parameter space is being searched
                if important_rad > IMP_RAD_CUTOFF:
                    print("\nInput:", input_names[p], "/ Output:", y_names[o], "- Neighbors not " \
                                                                               "found for specified n_neighbor threshold. Best attempt:",
                          len(filtered_neighbors))
                    break

                filtered_neighbors, debugger_matrix = get_neighbors(important, unimportant, X_normed, X_x0_normed,
                                                                    important_rad, unimportant_rad, x_not, p, o,
                                                                    debugger_matrix)

                # print statement, update ranges, check confounds
                if len(filtered_neighbors) >= n_neighbors:
                    neighbor_matrix, important_range, unimportant_range, confound_matrix = housekeeping(
                        neighbor_matrix, p, o, filtered_neighbors, verbose, input_names, y_names, important_rad,
                        unimportant_rad, important_range, unimportant_range, confound_matrix, X_x0_normed, X_normed,
                        important, unimportant)

                # if not enough neighbors are found, increment unimportant_radius until enough neighbors found
                # OR the radius is greater than important_radius*ratio
                if important_rad < .08:
                    upper_bound = 1.
                elif important_rad < .12:
                    upper_bound = 1.3
                else:
                    upper_bound = 1.7
                """elif important_rad < .22:
                    upper_bound = 2.2
                else:
                    upper_bound = 2.6"""
                upper_bound *= dominant_list[p]

                while len(filtered_neighbors) < n_neighbors and unimportant_rad < upper_bound:
                    filtered_neighbors, debugger_matrix = get_neighbors(important, unimportant, X_normed, X_x0_normed,
                                                                        important_rad, unimportant_rad, x_not, p, o,
                                                                        debugger_matrix)

                    if len(filtered_neighbors) >= n_neighbors:
                        neighbor_matrix, important_range, unimportant_range, confound_matrix = housekeeping(
                            neighbor_matrix, p, o, filtered_neighbors, verbose, input_names, y_names, important_rad,
                            unimportant_rad, important_range, unimportant_range, confound_matrix, X_x0_normed,
                            X_normed, important, unimportant)
                    unimportant_rad += UNIMP_RAD_INCREMENT

                important_rad += 10 ** magnitude

    print("Important independent variable radius range:", important_range, "/ Unimportant:", unimportant_range)
    return neighbor_matrix, confound_matrix, debugger_matrix


def get_coef(num_input, num_output, neighbor_matrix, X_normed, y_normed):
    """compute coefficients between parameter and feature based on linear regression. also get p-val
    coef will always refer to the R coefficient linear regression between param X and feature y

    :param num_input: int
    :param num_output: int
    :param neighbor_matrix: 2d array of lists which contain neighbor indices
    :param X_normed: 2d array of input vars normalized
    :param y_normed: 2d array of output vars normalized
    :return:
    """
    coef_matrix = np.zeros((num_input, num_output))
    pval_matrix = np.ones((num_input, num_output))

    for inp in range(num_input):
        for out in range(num_output):
            neighbor_array = neighbor_matrix[inp][out]
            if neighbor_array:
                selection = [ind for ind in neighbor_array]
                X_sub = X_normed[selection, inp]  # get relevant X data points

                coef_matrix[inp][out] = stats.linregress(X_sub, y_normed[selection, out])[2]
                pval_matrix[inp][out] = stats.linregress(X_sub, y_normed[selection, out])[3]

    return coef_matrix, pval_matrix


def determine_confounds(num_input, num_output, coef_matrix, pval_matrix, confound_matrix, input_names,
                        y_names, important_parameters, neighbor_matrix):
    """for each significant feature/parameter relationship identified, check if possible confounds are significant"""
    sig_confounds = np.zeros((num_input, num_output))
    P_BASELINE = .05

    # confound
    for param in range(num_input):
        for feat in range(num_output):
            if pval_matrix[param][feat] < .05 and confound_matrix[param][feat]:  # magic number
                for confound in confound_matrix[param][feat]:
                    if coef_matrix[confound][feat] > .03 and pval_matrix[confound][
                        feat] < P_BASELINE:  # magic number .03
                        print("Possible confound for cell", input_names[param], "/", y_names[feat], \
                              ":", input_names[confound], "with p-val", pval_matrix[confound][feat], "and coef", \
                              coef_matrix[confound][feat])
                        sig_confounds[param][feat] = 1

    # globally important, but locally not important (confound)
    for feat in range(num_output):
        important_parameter_set = important_parameters[feat]
        for param in important_parameter_set:  # param is a str
            param_index = np.where(input_names == param)[0][0]
            if sig_confounds[param_index][feat] != 1:
                sig_confounds[param_index][feat] = .6

    # not enough neighbors
    for param in range(num_input):
        for feat in range(num_output):
            if not neighbor_matrix[param][feat]:
                sig_confounds[param][feat] = .2
    return sig_confounds


def normalize_coef(num_input, num_output, coef_matrix, pval_matrix, p_baseline, sig_confounds):
    """normalize absolute coefficients by column. only normalize the ones less than the pval

    :param num_input: int
    :param num_output: int
    :param coef_matrix: 2d array (R coef)
    :param pval_matrix: 2d array
    :param p_baseline: float between 0 and 1
    :param sig_confounds: 2d array of floats
    :return:
    """
    coef_normed = abs(np.copy(coef_matrix))
    for output in range(num_output):
        sig_values = []
        for inp in range(num_input):
            if pval_matrix[inp][output] < p_baseline and sig_confounds[inp][output] == 0:
                sig_values.append(abs(coef_matrix[inp][output]))
        if sig_values:  # if no significant values for an objective, they won't be plotted anyway
            max_coef = np.amax(sig_values)
            min_coef = np.amin(sig_values)
            range_coef = max_coef - min_coef

            if range_coef == 0:
                coef_normed[:, output] = 1
            else:
                coef_normed[:, output] = np.true_divide((coef_normed[:, output] - min_coef), range_coef)

    return coef_normed


def plot_sensitivity(num_input, num_output, coef_matrix, pval_matrix, input_names, y_names, sig_confounds):
    """plot local sensitivity. mask cells with confounds and p-vals greater than than baseline
    color = sig, white = non-sig
    LGIHEST gray = no neighbors, light gray = confound but DT marked as important, dark gray = confound

    :param num_input: int
    :param num_output: int
    :param coef_matrix: 2d array of floats
    :param pval_matrix: 2d array of floats
    :param input_names: list of str
    :param y_names: list of str
    :param sig_confounds: 2d array of floats: 0 (no sig confound), .2 (no neighbors)
                          .6 (confound but marked imp by DT), or 1 (confound)
    :return:
    """
    import seaborn as sns
    P_BASELINE = .05

    # create mask
    mask = np.full((num_input, num_output), True, dtype=bool)  # mask
    mask[pval_matrix < P_BASELINE] = False  # do not mask
    mask[sig_confounds != 0] = True  # mask

    # overlay relationship heatmap (hm) with confound heatmap
    fig, ax = plt.subplots(figsize=(16, 5))
    hm = sns.heatmap(coef_matrix, fmt="g", cmap='cool', vmax=.3, vmin=0, mask=mask, linewidths=1, ax=ax)
    hm2 = sns.heatmap(sig_confounds, fmt="g", cmap='Greys', vmax=1, linewidths=1, ax=ax, alpha=.3, cbar=False)
    hm.set_xticklabels(y_names)
    hm.set_yticklabels(input_names)
    plt.xticks(rotation=-90)
    plt.yticks(rotation=0)
    plt.title("Absolute R Coefficients")
    plt.show()


def prompt_values():
    """initial prompt for variable values"""
    n_neighbors = 60
    max_dist = .01

    user_input = input('Do you want to specify the values for neighbor search? The default values are num '
                            'neighbors = 60, and starting radius for important independent variables = .01. (y/n) ')
    if user_input in ['y', 'Y']:
        n_neighbors = int(input('Threshold for number of neighbors?: '))
        max_dist = float(input('Starting radius for important independent variables?: '))
    elif user_input in ['n', 'N']:
        print('Thanks.')
    else:
        while user_input not in ['y', 'Y', 'n', 'N']:
            user_input = input('Please enter y or n. ')

    return n_neighbors, max_dist


def prompt_neighbor_dialog(num_input, num_output, num_param, important_inputs, input_names, y_names, X_normed,
                           x0_normed, verbose, n_neighbors, max_dist, input_is_not_param, inp_out_same, dominant_list):
    """at the end of neighbor search, ask the user if they would like to change the starting variables"""
    while True:
        neighbor_matrix, confound_matrix, debugger_matrix = compute_neighbor_matrix(num_input, num_output, num_param,
                                                                                    important_inputs, input_names,
                                                                                    y_names, X_normed, x0_normed,
                                                                                    verbose, n_neighbors, max_dist,
                                                                                    input_is_not_param, inp_out_same,
                                                                                    dominant_list)
        user_input = ''
        while user_input.lower() not in ['y', 'n', 'yes', 'no']:
            user_input = input('Was this an acceptable outcome (y/n)? ')
        if user_input.lower() in ['y', 'yes']:
            break
        elif user_input.lower() in ['n', 'no']:
            n_neighbors, max_dist = prompt_values()

    return neighbor_matrix, confound_matrix, debugger_matrix


def denormalize(scaling, unnormed_vector, param, logdiff_array, logmin_array, diff_array, min_array):
    if scaling[param] == 'log':
        unnormed_vector = np.power(10, (unnormed_vector * logdiff_array[param] + logmin_array[param]))
    else:
        unnormed_vector = unnormed_vector * diff_array[param] + min_array[param]

    return unnormed_vector


def create_perturb_matrix(X_best, n_neighbors, input, perturbations):
    """
    :param X_best: x0
    :param n_neighbors: int, how many perturbations were made
    :param input: int, idx for independent variable to manipulate
    :param perturbations: array
    :return:
    """
    perturb_matrix = np.tile(np.array(X_best), (n_neighbors, 1))
    perturb_matrix[:, input] = perturbations

    return perturb_matrix


def generate_explore_vector(n_neighbors, num_input, num_output, X_best, X_x0_normed, scaling, logdiff_array,
                            logmin_array, diff_array, min_array, neighbor_matrix, norm_search):
    """
    figure out which X/y pairs need to be explored: non-sig or no neighbors
    generate n_neighbor points around best point. perturb just POI... 5% each direction

    :return: dict, key=param number (int), value=list of arrays
    """
    explore_dict = {}

    # if n_neighbors is odd
    if n_neighbors % 2 == 1:
        n_neighbors += 1

    for inp in range(num_input):
        for output in range(num_output):
            if neighbor_matrix[inp][output] is not None and len(neighbor_matrix[inp][output]) < n_neighbors:
                upper = .05 * np.random.random_sample((int(n_neighbors / 2),)) + X_x0_normed[inp]
                lower = .05 * np.random.random_sample((int(n_neighbors / 2),)) + X_x0_normed[inp] - .05
                unnormed_vector = np.concatenate((upper, lower), axis=0)

                perturbations = unnormed_vector if not norm_search else denormalize(
                    scaling, unnormed_vector, inp, logdiff_array, logmin_array, diff_array, min_array)
                perturb_matrix = create_perturb_matrix(X_best, n_neighbors, inp, perturbations)
                explore_dict[inp] = perturb_matrix
                break

    return explore_dict


def save_perturbation_PopStorage(perturb_dict, param_id2name, save_path=''):
    import time
    full_path = save_path + 'perturbations_%i_%i_%i.h5' % (
    time.localtime()[2], time.localtime()[3], time.localtime()[-4])
    with h5py.File(full_path, 'a') as f:
        for param_id in perturb_dict:
            param = param_id2name[param_id]
            f.create_group(param)
            for i in range(len(perturb_dict[param_id])):
                f[param][str(i)] = perturb_dict[param_id][i]


def convert_dict_to_PopulationStorage(explore_dict, input_names, output_names, obj_names, save_path=''):
    """unsure if storing in PS object is needed; save function only stores array"""
    pop = PopulationStorage(param_names=input_names, feature_names=output_names, objective_names=obj_names,
                            path_length=1, file_path=None)
    iter_to_param_map = {}
    for i, param_id in enumerate(explore_dict):
        iter_to_param_map[i] = input_names[param_id]
        iteration = []
        for vector in explore_dict[param_id]:
            indiv = Individual(vector)
            indiv.objectives = []
            iteration.append(indiv)
        pop.append(iteration)
    save_perturbation_PopStorage(explore_dict, input_names, save_path)
    return iter_to_param_map, pop


def prompt_indiv(storage):
    fame = HallOfFame(storage)
    user_input = ''
    while user_input not in fame.x_dict:
        print('Valid strings for x0: ', list(fame.x_dict.keys()))
        user_input = input('Specify x0: ')

    return user_input


def prompt_feat_or_obj():
    user_input = ''
    while user_input.lower() not in ['f', 'o', 'features', 'objectives', 'feature', 'objective', 'feat', 'obj']:
        user_input = input('Do you want to analyze features or objectives?: ')
    return user_input.lower() in ['f', 'features', 'feature', 'feat']


def prompt_linspace():
    user_input = ''
    while user_input.lower() not in ['y', 'n', 'yes', 'no']:
        user_input = input('Normalize the data?: ')
    return user_input.lower() in ['y', 'yes']


def prompt_no_LSA():
    user_input = ''
    while user_input.lower() not in ['y', 'n', 'yes', 'no']:
        user_input = input('Do you just want to simply plot input vs. output without filtering (no LSA)?: ')
    return user_input.lower() in ['y', 'yes']


def prompt_input():
    user_input = ''
    while user_input.lower() not in ['f', 'o', 'feature', 'objective', 'parameter', 'p', 'features', 'objectives',
                                     'parameters']:
        user_input = input('What is the independent variable (features/objectives/parameters)?: ')
    return user_input.lower()


def prompt_output():
    user_input = ''
    while user_input.lower() not in ['f', 'o', 'feature', 'objective', 'features', 'objectives']:
        user_input = input('What is the the dependent variable (features/objectives)?: ')
    return user_input.lower()


def prompt_DT_constraint():
    user_input = ''
    while user_input.lower() not in ['y', 'n', 'yes', 'no']:
        user_input = input('During neighbor search, should the constraint for unimportant input variables be relaxed '
                  'if the magnitude of the mean of the feature importance of the important variables is '
                  'twice or more that of the unimportant variables?: ')
    return user_input.lower() in ['y', 'yes']


def prompt_relax_constraint():
    user_input = ''
    while user_input is not float:
        try:
            user_input = float(input('By what factor should it be relaxed? The default is 1.5: '))
            return float(user_input)
        except ValueError:
            print('Please enter a number.')
    return 1.5


def get_variable_names(population, input_str, output_str, obj_strings, feat_strings, param_strings):
    if input_str in obj_strings:
        input_names = population.objective_names
    elif input_str in feat_strings:
        input_names = population.feature_names
    elif input_str in param_strings:
        input_names = population.param_names
    else:
        raise RuntimeError('LSA: input variable %s is not recognized' % input_str)

    if output_str in obj_strings:
        y_names = population.objective_names
    elif output_str in feat_strings:
        y_names = population.feature_names
    else:
        raise RuntimeError('LSA: output variable %s is not recognized' % output_str)
    return input_names, y_names


def local_sensitivity(population, verbose=True, save_path=''):
    """main function for plotting and computing local sensitivity
    note on variable names: X_x0 redundantly refers to the parameter values associated with the point x0. x0 by itself
    refers to both the parameters and the output
    input = independent var, output = dependent var

    :param population: PopulationStorage object
    :param verbose: bool. if True, will print radius and num neighbors for each parameter/objective pair
    :param save_path: str for where perturbation vector will be saved if generated
    :return:
    """
    feat_strings = ['f', 'feature', 'features']
    obj_strings = ['o', 'objective', 'objectives']
    param_strings = ['parameter', 'p', 'parameters']

    x0_string = prompt_indiv(population)
    input_str = prompt_input()
    output_str = prompt_output()
    feat_bool = output_str in feat_strings
    no_LSA = prompt_no_LSA()
    relaxed_bool = prompt_DT_constraint() if not no_LSA else False
    relaxed_factor = prompt_relax_constraint() if relaxed_bool else 1.
    norm_search = prompt_linspace()

    data = pop_to_matrix(population, feat_bool)
    processed_data, crossing, z = process_data(data)

    input_names, y_names = get_variable_names(population, input_str, output_str, obj_strings, feat_strings,
                                              param_strings)
    num_param = len(population.param_names)
    num_input = len(input_names)
    num_output = len(y_names)
    input_is_not_param = input_str not in param_strings
    inp_out_same = (input_str in feat_strings and output_str in feat_strings) or \
                   (input_str in obj_strings and output_str in obj_strings)

    data_normed, x0_normed, packaged_variables = normalize_data(population, data, processed_data, crossing, z, x0_string,
            population.param_names, input_is_not_param, norm_search)
    if no_LSA:
        lsa_obj = LSA(None, None, None, None, input_names, y_names, data_normed)
        print("No exploration vector generated.")
        return None, lsa_obj, None

    X_x0 = packaged_variables[0];
    scaling = packaged_variables[1];
    logdiff_array = packaged_variables[2]
    logmin_array = packaged_variables[3];
    diff_array = packaged_variables[4];
    min_array = packaged_variables[5]

    X_normed = data_normed[:, :num_param] if input_str in param_strings else data_normed[:, num_param:]
    y_normed = data_normed[:, num_param:]

    important_inputs, dominant_list = get_important_inputs2(data_normed, num_input, num_output, num_param, input_names, y_names,
                                            input_is_not_param, inp_out_same, relaxed_factor)

    n_neighbors, max_dist = prompt_values()
    neighbor_matrix, confound_matrix, debugger_matrix = prompt_neighbor_dialog(num_input, num_output, num_param,
                                                                               important_inputs,
                                                                               input_names, y_names, X_normed,
                                                                               x0_normed, verbose, n_neighbors,
                                                                               max_dist, input_is_not_param,
                                                                               inp_out_same,
                                                                               dominant_list)

    coef_matrix, pval_matrix = get_coef(num_input, num_output, neighbor_matrix, X_normed, y_normed)
    sig_confounds = determine_confounds(num_input, num_output, coef_matrix, pval_matrix, confound_matrix,
                                        input_names, y_names, important_inputs, neighbor_matrix)

    if input_is_not_param:
        explore_pop = None
    else:
        explore_dict = generate_explore_vector(n_neighbors, num_input, num_output, X_x0, x0_normed[:num_input],
                                               scaling, logdiff_array, logmin_array, diff_array, min_array,
                                               neighbor_matrix, norm_search)
        explore_pop = convert_dict_to_PopulationStorage(explore_dict, input_names, population.feature_names,
                                                        population.objective_names, save_path)

    plot_sensitivity(num_input, num_output, coef_matrix, pval_matrix, input_names, y_names, sig_confounds)
    lsa_obj = LSA(neighbor_matrix, coef_matrix, pval_matrix, sig_confounds, input_names, y_names, data_normed)
    debug = DebugObject(debugger_matrix, data_normed, input_names, y_names, important_inputs)
    if input_is_not_param:
        print("The exploration vector for the parameters was not generated because it was not the dependent variable.")
    return explore_pop, lsa_obj, debug


class LSA(object):
    def __init__(self, neighbor_matrix, coef_matrix, pval_matrix, sig_confounds, input_id2name, y_id2name, data):
        self.neighbor_matrix = neighbor_matrix
        self.coef_matrix = coef_matrix
        self.pval_matrix = pval_matrix
        self.sig_confounds = sig_confounds
        self.data = data
        self.input_name2id = {}
        self.y_name2id = {}

        for i, name in enumerate(input_id2name): self.input_name2id[name] = i
        for i, name in enumerate(y_id2name): self.y_name2id[name] = i

    def plot_indep_vs_dep(self, input_name, y_name, use_unfiltered_data=False, num_models=None, last_third=True):
        input_id = self.input_name2id[input_name]
        y_id = self.y_name2id[y_name]
        if self.neighbor_matrix is None: use_unfiltered_data = True
        if not use_unfiltered_data:
            neighbor_indices = self.neighbor_matrix[input_id][y_id]
            if neighbor_indices is None:
                print("No neighbors-- nothing to show.")
            else:
                x = self.data[neighbor_indices, input_id]
                y = self.data[neighbor_indices, y_id]
                plt.scatter(x, y)
                fit_fn = np.poly1d(np.polyfit(x, y, 1))
                plt.plot(x, fit_fn(x), color='red')

                if self.sig_confounds[input_id][y_id] == .6:
                    plt.title("%s vs %s with p-val of %.3f and R coef of %.3f. Locally confounded "
                              "but deemed globally important" % (input_name, y_name, self.pval_matrix[input_id][y_id],
                                                                 self.coef_matrix[input_id][y_id]))
                elif self.sig_confounds[input_id][y_id] == 1:
                    plt.title("%s vs %s with p-val of %.3f and R coef of %.3f, Locally confounded and "
                              "not deemed globally important" % (input_name, y_name, self.pval_matrix[input_id][y_id],
                                                                 self.coef_matrix[input_id][y_id]))
                else:
                    plt.title("%s vs %s with p-val of %.3f and R coef of %.3f. Not confounded" % \
                              (input_name, y_name, self.pval_matrix[input_id][y_id], self.coef_matrix[input_id][y_id]))
        else:
            if num_models is not None:
                num_models = int(num_models)
                x = self.data[-num_models:, input_id]
                y = self.data[-num_models:, y_id]
                plt.scatter(x, y, c=np.arange(self.data.shape[0] - num_models, self.data.shape[0]), cmap='viridis_r')
                plt.title("Last %i models" % num_models)
            elif last_third:
                m = int(self.data.shape[0] / 3)
                x = self.data[-m:, input_id]
                y = self.data[-m:, y_id]
                plt.scatter(x, y, c=np.arange(self.data.shape[0] - m, self.data.shape[0]), cmap='viridis_r')
                plt.title("Last third of models")
            else:
                x = self.data[:, input_id]
                y = self.data[:, y_id]
                plt.scatter(x, y, c=np.arange(self.data.shape[0]), cmap='viridis_r')
                plt.title("All models")
            plt.colorbar()

        plt.xlabel(input_name)
        plt.ylabel(y_name)
        plt.show()


class DebugObject(object):
    """
    debug plotter - after simulation
    one per i/o pair -> if not None, if more than 10% of the space is being searched or more than 500, whichever is less
    remember params

    split by:
    -passed unimp filter
    -passed imp filter (unsig)
    -passed imp + sig filter
    -passed both distance-based filters
    -passed all constraints
    """
    def __init__(self, debug_matrix, data, input_id2name, y_id2name, important_inputs):
        """

        :param debug_matrix: actually a dict (key=input id) of dicts (key=output id) of lists of tuples of the form
        (array representing point in input space, string representing category)
        :param y_id2name:
        """
        self.debug_matrix = debug_matrix
        self.data = data
        self.input_id2name = input_id2name
        self.important_inputs = important_inputs
        self.input_name2id = {}
        self.y_name2id = {}
        self.cat2color = {'UI': 'red', 'I': 'blue', 'DIST': 'purple', 'SIG': 'green', 'ALL': 'cyan'}
        self.previous_plot_data = defaultdict(dict)

        for i, name in enumerate(input_id2name): self.input_name2id[name] = i
        for i, name in enumerate(y_id2name): self.y_name2id[name] = i

    def get_points(self, input_name, y_name):
        try:
            buckets = self.debug_matrix[self.input_name2id[input_name]][self.y_name2id[y_name]]
        except:
            raise RuntimeError(
                'At least one provided variable name is incorrect. For input variables, valid choices are ',
                list(self.input_name2id.keys()), '. For output variables, ', list(self.y_name2id.keys()), '.')
        return buckets

    def extract_data(self, input_name, y_name):
        if input_name in self.previous_plot_data and y_name in self.previous_plot_data[input_name]:
            all_points = self.previous_plot_data[input_name][y_name][0]
            cat2idx = self.previous_plot_data[input_name][y_name][1]
        else:
            buckets = self.get_points(input_name, y_name)
            all_points = None
            cat2idx = defaultdict(list)
            idx_counter = 0
            for cat, idx in buckets.items():
                idx_list = list(idx) #idx is a set
                if len(idx_list) == 0: continue
                all_points = self.data[idx_list] if all_points is None else np.concatenate((all_points, self.data[idx_list]))
                cat2idx[cat] = list(range(idx_counter, idx_counter + len(idx_list)))
                idx_counter += len(idx_list)
            self.previous_plot_data[input_name][y_name] = (all_points, cat2idx)
        return all_points, cat2idx

    def plot_PCA(self, input_name, y_name):
        """try visualizing all of the input variable values by flattening it"""
        all_points, cat2idx = self.extract_data(input_name, y_name)
        if all_points is not None:
            pca = PCA(n_components=2)
            pca.fit(all_points)
            flattened = pca.transform(all_points)

            for cat in self.cat2color:
                idxs = cat2idx[cat]
                plt.scatter(flattened[idxs, 0], flattened[idxs, 1], c=self.cat2color[cat], label=cat, alpha=.3)
            plt.legend(labels=list(self.cat2color.keys()))
            plt.xlabel('Principal component 1 (%.3f)' % pca.explained_variance_ratio_[0])
            plt.ylabel('Principal component 2 (%.3f)' % pca.explained_variance_ratio_[1])
            plt.title('Neighbor search for the sensitivity of %s to %s' % (y_name, input_name))
            plt.show()
        else:
            print("No neighbors-- nothing to show.")

    def plot_vs(self, input_name, y_name, x1, x2):
        """plot one input variable vs another input"""
        try:
            x1_idx = self.input_name2id[x1]
            x2_idx = self.input_name2id[x2]
        except:
            raise RuntimeError(
                'At least one provided variable name is incorrect. For input variables, valid choices are ',
                list(self.input_name2id.keys()), '.')

        all_points, cat2idx = self.extract_data(input_name, y_name)
        for cat in self.cat2color:
            idxs = cat2idx[cat]
            plt.scatter(all_points[idxs, x1_idx], all_points[idxs, x2_idx], c=self.cat2color[cat], label=cat, alpha=.3)
        plt.legend(labels=list(self.cat2color.keys()))
        plt.xlabel(x1)
        plt.ylabel(x2)
        plt.title('Neighbor search for the sensitivity of %s to %s' % (y_name, input_name))
        plt.show()

    def get_interference_by_classification(self, input_name, y_name):
        all_points, cat2idx = self.extract_data(input_name, y_name)
        if all_points is None:
            print('No neighbors found.')
        else:
            y_labels = np.zeros(all_points.shape[0])
            for idx in cat2idx['ALL']: y_labels[idx] = 1

            if np.all(y_labels == 0):
                print('Could not calculate interference; no points were accepted by the filter.')
            else:
                dt = DecisionTreeClassifier(random_state=0, max_depth=200)
                dt.fit(all_points, y_labels)

                input_list = list(zip([round(t, 4) for t in dt.feature_importances_], list(self.input_name2id.keys())))
                input_list.sort(key=lambda x: x[0], reverse=True)
                print('The top five input variables that interfered (based on Gini importance) were: ', input_list[:5])

    def get_interference_manually(self, input_name, y_name):
        all_points, cat2idx = self.extract_data(input_name, y_name)
        if all_points is None:
            print('No neighbors found.')
        else:
            print('Out of %d points that passed the important distance filter, %d had signficant perturbations in the '
                  'direction of %s' % (all_points.shape[0] - len(cat2idx['UI']), (len(cat2idx['SIG']) + len(cat2idx['ALL'])),
                  input_name))

            count_arr = np.zeros((len(self.input_name2id), 1))
            y_idx = self.y_name2id[y_name]
            # pass unimp filter, but not imp
            """for idx in cat2idx['SIG']:
                print(all_points[idx, self.important_inputs[y_idx]])
                max_idx = np.argmax(all_points[idx, important_idx])
                count_arr[max_idx] += 1"""

            # pass imp but not unimp
            important_idx = [self.input_name2id[key] for key in self.important_inputs[y_idx]]
            for cat_name in ['SIG', 'I']:
                for idx in cat2idx[cat_name]:
                    tmp_idx = [x for x in range(all_points.shape[1]) if x not in important_idx \
                               and x in list(range(len(self.input_name2id)))]
                    max_idx = np.argmax(all_points[idx, tmp_idx])
                    count_arr[max_idx] += 1

            ratios = count_arr / np.sum(count_arr)
            sort_idx = np.argsort(-ratios, axis=0)  #descending order
            sorted_ratios = sorted(ratios, reverse=True)
            for i in range(len(sorted_ratios)):
                j = np.where(sort_idx == i)[0][0]
                print('%s: %.3f' % (self.input_id2name[j], sorted_ratios[i]))
