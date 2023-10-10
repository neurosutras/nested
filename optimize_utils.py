"""
Library of functions and classes to support nested.optimize
"""
__author__ = 'Aaron D. Milstein, Grace Ng, and Prannath Moolchand'

from nested.utils import *
from nested.parallel import find_context, find_context_name
import collections
from scipy._lib._util import check_random_state
from copy import deepcopy
import uuid
import warnings
import shutil


class Individual(object):
    """

    """

    def __init__(self, x, model_id=None):
        """

        :param x: array
        """
        self.x = np.array(x)
        self.features = None
        self.objectives = None
        self.normalized_objectives = None
        self.energy = None
        self.rank = None
        self.distance = None
        self.fitness = None
        self.survivor = False
        self.model_id = model_id


class OptimizationHistory(object):
    """
    Class used to store populations of parameters and objectives during optimization.
    """

    def __init__(self, param_names=None, feature_names=None, objective_names=None, path_length=None,
                 normalize='global', file_path=None):
        """

        :param param_names: list of str
        :param feature_names: list of str
        :param objective_names: list of str
        :param path_length: int
        :param normalize: str; 'global': normalize over entire history, 'local': normalize per iteration
        :param file_path: str (path)
        """
        if file_path is not None:
            from nested.lsa import sum_objectives
            if os.path.isfile(file_path):
                self.load(file_path)
            else:
                raise IOError('OptimizationHistory: invalid file path: %s' % file_path)
            self.total_models = sum([len(gen) for gen in self.generations])  # doesn't include failed models
            self.summed_obj = sum_objectives(self, self.total_models)  # for plotting
            self.best_model = self.survivors[-1][0] if self.survivors and self.survivors[-1] else None
            self.param_matrix, self.obj_matrix, self.feat_matrix = [None] * 3  # for dumb_plot
        else:
            if not isinstance(param_names, collections.Iterable):
                raise TypeError('OptimizationHistory: param_names must be specified as a list of str')
            self.param_names = param_names
            if feature_names is not None and not isinstance(feature_names, collections.Iterable):
                raise TypeError('OptimizationHistory: feature_names must be specified as a list of str')
            self.feature_names = feature_names
            if objective_names is not None and not isinstance(objective_names, collections.Iterable):
                raise TypeError('OptimizationHistory: objective_names must be specified as a list of str')
            self.objective_names = objective_names

            if type(path_length) == int:
                self.path_length = path_length
            else:
                raise TypeError('OptimizationHistory: path_length must be specified as int')
            if normalize in ['local', 'global']:
                self.normalize = normalize
            else:
                raise ValueError('OptimizationHistory: normalize argument must be either \'global\' or \'local\'')
            self.generations = []  # a list of populations, each corresponding to one generation
            self.survivors = []  # a list of populations (some may be empty)
            self.specialists = []  # a list of populations (some may be empty)
            self.prev_survivors = []  # a list of populations (some may be empty)
            self.prev_specialists = []  # a list of populations (some may be empty)
            self.failed = []  # a list of populations (some may be empty)
            self.min_objectives = []  # list of array of float
            self.max_objectives = []  # list of array of float
            # Enable tracking of user-defined attributes through kwargs to 'append'
            self.attributes = {}
            self.count = 0
            self.total_models = 0  # total_models != count

    def append(self, population, survivors=None, specialists=None, prev_survivors=None,
               prev_specialists=None, failed=None, min_objectives=None, max_objectives=None, **kwargs):
        """

        :param population: list of :class:'Individual'
        :param survivors: list of :class:'Individual'
        :param specialists: list of :class:'Individual'
        :param prev_survivors: list of :class:'Individual'
        :param prev_specialists: list of :class:'Individual'
        :param failed: list of :class:'Individual'
        :param min_objectives: array of float
        :param max_objectives: array of float
        :param kwargs: dict of additional param_gen-specific attributes
        """
        if survivors is None:
            survivors = []
        if specialists is None:
            specialists = []
        if prev_survivors is None:
            prev_survivors = []
        if prev_specialists is None:
            prev_specialists = []
        if failed is None:
            failed = []
        if min_objectives is None:
            min_objectives = []
        if max_objectives is None:
            max_objectives = []
        self.survivors.append(deepcopy(survivors))
        self.specialists.append(deepcopy(specialists))
        self.prev_survivors.append(deepcopy(prev_survivors))
        self.prev_specialists.append(deepcopy(prev_specialists))
        self.generations.append(deepcopy(population))
        self.failed.append(deepcopy(failed))
        self.count += len(population) + len(failed)
        self.total_models += len(population)
        self.min_objectives.append(deepcopy(min_objectives))
        self.max_objectives.append(deepcopy(max_objectives))

        for key in kwargs:
            if key not in self.attributes:
                self.attributes[key] = []
        for key in self.attributes:
            if key in kwargs:
                self.attributes[key].append(kwargs[key])
            else:
                self.attributes[key].append(None)

    def plot(self, subset=None, show_failed=False, mark_specialists=True, energy_scale='log', energy_color='relative'):
        """

        :param subset: can be str, list, or dict
            valid categories: 'features', 'objectives', 'parameters'
            valid dict vals: list of str of valid category names
        :param show_failed: bool; whether to show failed models when plotting parameters
        :param mark_specialists: bool; whether to mark specialists
        :param energy_scale: str in ['log', 'linear']; how to scale relative and objective error in plots
        :param energy_color: str in ['relative','absolute']; how to color points when plotting categories
        """
        def get_group_stats(groups):
            """

            :param groups: defaultdict(list(list of float))
            :return: tuple of array
            """
            min_vals = []
            max_vals = []
            median_vals = []
            for i in range(len(next(iter(groups.values())))):
                vals = []
                for group_name in groups:
                    vals.extend(groups[group_name][i])
                min_vals.append(np.min(vals))
                max_vals.append(np.max(vals))
                median_vals.append(np.median(vals))
            min_val = np.min(min_vals)
            max_val = np.max(max_vals)
            median_vals = np.array(median_vals)

            return min_val, max_val, median_vals

        import matplotlib.pyplot as plt
        from matplotlib.pyplot import cm
        import matplotlib as mpl
        from matplotlib.lines import Line2D
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        mpl.rcParams['svg.fonttype'] = 'none'
        mpl.rcParams['text.usetex'] = False
        cmap = cm.viridis # cm.rainbow
        cmap_name = 'viridis'

        default_categories = {'parameters': self.param_names, 'objectives': self.objective_names,
                              'features': self.feature_names}
        if subset is None:
            categories = default_categories
        elif isinstance(subset, (str, bytes)):
            if subset not in default_categories:
                raise KeyError('OptimizationHistory.plot: invalid category provided to subset argument: %s' % subset)
            else:
                categories = {subset: default_categories[subset]}
        elif isinstance(subset, list):
            categories = dict()
            for key in subset:
                if key not in default_categories:
                    raise KeyError('OptimizationHistory.plot: invalid category provided to subset argument: %s' % key)
                categories[key] = default_categories[key]
        elif isinstance(subset, dict):
            for key in subset:
                if key not in default_categories:
                    raise KeyError('OptimizationHistory.plot: invalid category provided to subset argument: %s' % key)
                if not isinstance(subset[key], list):
                    raise ValueError('OptimizationHistory.plot: subset category names must be provided as a list')
                valid_elements = default_categories[key]
                for element in subset[key]:
                    if element not in valid_elements:
                        raise KeyError('OptimizationHistory.plot: invalid %s name provided to subset argument: %s' %
                                       (key[:-1], element))
            categories = subset
        else:
            raise ValueError('OptimizationHistory.plot: invalid type of subset argument')

        ranks_history = defaultdict(list)
        fitness_history = defaultdict(list)
        rel_energy_history = defaultdict(list)
        abs_energy_history = defaultdict(list)
        param_history = defaultdict(lambda: defaultdict(list))
        feature_history = defaultdict(lambda: defaultdict(list))
        objective_history = defaultdict(lambda: defaultdict(list))
        param_name_list = self.param_names
        feature_name_list = self.feature_names
        objective_name_list = self.objective_names

        max_gens = len(self.generations)
        num_gen = 0
        max_iter = 0
        while num_gen < max_gens:
            this_iter_specialist_ids = \
                set([individual.model_id for individual in self.specialists[num_gen + self.path_length - 1]])
            groups = defaultdict(list)
            for i in range(self.path_length):
                this_gen = list(set(self.prev_survivors[num_gen + i] + self.prev_specialists[num_gen + i]))
                this_gen.extend(self.generations[num_gen + i])
                for individual in this_gen:
                    if mark_specialists and individual.model_id in this_iter_specialist_ids:
                        groups['specialists'].append(individual)
                    elif individual.survivor:
                        groups['survivors'].append(individual)
                    else:
                        groups['population'].append(individual)
                groups['failed'].extend(self.failed[num_gen + i])

            for group_name in ['population', 'survivors', 'specialists']:
                group = groups[group_name]
                this_ranks = []
                this_fitness = []
                this_rel_energy = []
                this_abs_energy = []
                for individual in group:
                    this_ranks.append(individual.rank)
                    this_fitness.append(individual.fitness)
                    this_rel_energy.append(individual.energy)
                    this_abs_energy.append(np.sum(individual.objectives))
                ranks_history[group_name].append(this_ranks)
                fitness_history[group_name].append(this_fitness)
                rel_energy_history[group_name].append(this_rel_energy)
                abs_energy_history[group_name].append(this_abs_energy)
                if 'parameters' in categories:
                    for param_name in categories['parameters']:
                        index = param_name_list.index(param_name)
                        this_param_history = []
                        for individual in group:
                            this_param_history.append(individual.x[index])
                        param_history[param_name][group_name].append(this_param_history)
                if 'features' in categories:
                    for feature_name in categories['features']:
                        index = feature_name_list.index(feature_name)
                        this_feature_history = []
                        for individual in group:
                            this_feature_history.append(individual.features[index])
                        feature_history[feature_name][group_name].append(this_feature_history)
                if 'objectives' in categories:
                    for objective_name in categories['objectives']:
                        index = objective_name_list.index(objective_name)
                        this_objective_history = []
                        for individual in group:
                            this_objective_history.append(individual.objectives[index])
                        objective_history[objective_name][group_name].append(this_objective_history)

            if 'parameters' in categories:
                group_name = 'failed'
                group = groups[group_name]
                for param_name in categories['parameters']:
                    index = param_name_list.index(param_name)
                    this_param_history = []
                    for individual in group:
                        this_param_history.append(individual.x[index])
                    param_history[param_name][group_name].append(this_param_history)

            num_gen += self.path_length
            max_iter += 1

        fitness_min, fitness_max, fitness_med = get_group_stats(fitness_history)

        fig, axes = plt.subplots(1, figsize=(7., 4.8))
        norm = mpl.colors.Normalize(vmin=fitness_min-0.5, vmax=fitness_max+0.5)
        for i in range(max_iter):
            axes.scatter(np.ones(len(ranks_history['population'][i])) * (i + 1), ranks_history['population'][i],
                         c=fitness_history['population'][i],  # this_colors,
                         cmap=cmap, norm=norm, alpha=0.2, s=5., linewidth=0)
            axes.scatter(np.ones(len(ranks_history['specialists'][i])) * (i + 1), ranks_history['specialists'][i],
                         c=fitness_history['specialists'][i],  # this_colors,
                         cmap=cmap, norm=norm, alpha=0.2, s=5., linewidth=0)
            axes.scatter(np.ones(len(ranks_history['survivors'][i])) * (i + 1), ranks_history['survivors'][i],
                         c=fitness_history['survivors'][i],  # this_colors,
                         cmap=cmap, norm=norm, alpha=0.3, s=10., linewidth=0)
        axes.set_xlabel('Number of iterations')
        axes.set_ylabel('Model rank')
        axes.set_title('Fitness')
        box = axes.get_position()
        axes.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
        divider = make_axes_locatable(axes)
        cax = divider.append_axes('right', size='3%', pad=0.1)
        cbar = mpl.colorbar.ColorbarBase(cax, cmap=cm.get_cmap(cmap_name, int(fitness_max + 1)), norm=norm,
                                         orientation='vertical')
        cbar.set_label('Fitness', rotation=-90)
        tick_interval = max(1, fitness_max // 5)
        cbar.set_ticks(list(range(0, int(fitness_max + 1), tick_interval)))
        cbar.ax.get_yaxis().labelpad = 15
        clean_axes(axes)
        fig.show()

        rel_energy_min, rel_energy_max, rel_energy_med = get_group_stats(rel_energy_history)

        fig, axes = plt.subplots(1, figsize=(7., 4.8))
        for i in range(max_iter):
            axes.scatter(np.ones(len(rel_energy_history['population'][i])) * (i + 1),
                         rel_energy_history['population'][i], c='salmon', edgecolor='none', alpha=0.2, s=5.)
            if mark_specialists:
                axes.scatter(np.ones(len(rel_energy_history['specialists'][i])) * (i + 1),
                             rel_energy_history['specialists'][i], c='salmon', edgecolor='k', linewidth=0.75,
                             alpha=0.5, s=10.)
            else:
                axes.scatter(np.ones(len(rel_energy_history['specialists'][i])) * (i + 1),
                             rel_energy_history['specialists'][i], c='salmon', edgecolor='none', alpha=0.2, s=5.)
            axes.scatter(np.ones(len(rel_energy_history['survivors'][i])) * (i + 1),
                         rel_energy_history['survivors'][i], c='b', edgecolor='none', alpha=0.3, s=10.)
        axes.plot(range(1, max_iter + 1), rel_energy_med, c='r')
        legend_elements = [Line2D([0], [0], color='r', lw=2, label='Median'),
                           Line2D([0], [0], marker='o', color='b', label='Survivors', markerfacecolor='b',
                                  markersize=5, markeredgewidth=0., linewidth=0, alpha=0.8)]
        if mark_specialists:
            legend_elements.append(Line2D([0], [0], marker='o', color='k', label='Specialists',
                                          markerfacecolor='none', markersize=5, markeredgewidth=1.25, linewidth=0,
                                          alpha=1.))
        axes.set_xlabel('Number of iterations')
        axes.set_ylabel('Multi-objective error score')
        axes.set_title('Multi-objective error score')
        box = axes.get_position()
        axes.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
        axes.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.2),
                    fancybox=True, frameon=False, ncol=len(legend_elements))
        clean_axes(axes)
        # fig.subplots_adjust(right=0.8)
        fig.show()

        abs_energy_min, abs_energy_max, abs_energy_med = get_group_stats(abs_energy_history)

        fig, axes = plt.subplots(1, figsize=(7., 4.8))
        for i in range(max_iter):
            axes.scatter(np.ones(len(abs_energy_history['population'][i])) * (i + 1),
                         abs_energy_history['population'][i], c='salmon', edgecolor='none', alpha=0.2, s=5.)
            if mark_specialists:
                axes.scatter(np.ones(len(abs_energy_history['specialists'][i])) * (i + 1),
                             abs_energy_history['specialists'][i], c='salmon', edgecolor='k', linewidth=0.75,
                             alpha=0.5, s=10.)
            else:
                axes.scatter(np.ones(len(abs_energy_history['specialists'][i])) * (i + 1),
                             abs_energy_history['specialists'][i], c='salmon', edgecolor='none', alpha=0.2, s=5.)
            axes.scatter(np.ones(len(abs_energy_history['survivors'][i])) * (i + 1),
                         abs_energy_history['survivors'][i], c='b', edgecolor='none', alpha=0.3, s=10.)
        if energy_scale == 'log':
            if abs_energy_min > 0.:
                axes.semilogy(range(1, max_iter + 1), abs_energy_med, c='r')
                axes.set_ylabel('Total objective error (log scale)')
            else:
                axes.plot(range(1, max_iter + 1), abs_energy_med, c='r')
                axes.set_ylabel('Total objective error')
        elif energy_scale == 'linear':
            axes.plot(range(1, max_iter + 1), abs_energy_med, c='r')
            axes.set_ylabel('Total objective error')
        else:
            raise RuntimeError('OptimizationHistory.plot: energy_scale must be either \'linear\' or \'log\'')
        legend_elements = [Line2D([0], [0], color='r', lw=2, label='Median'),
                           Line2D([0], [0], marker='o', color='b', label='Survivors', markerfacecolor='b',
                                  markersize=5, markeredgewidth=0., linewidth=0, alpha=0.8)]
        if mark_specialists:
            legend_elements.append(Line2D([0], [0], marker='o', color='k', label='Specialists',
                                          markerfacecolor='none', markersize=5, markeredgewidth=1.25, linewidth=0,
                                          alpha=1.))
        axes.set_xlabel('Number of iterations')
        axes.set_title('Total objective error')
        box = axes.get_position()
        axes.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
        axes.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.2),
                    fancybox=True, frameon=False, ncol=len(legend_elements))
        clean_axes(axes)
        fig.show()

        if energy_color == 'relative':
            norm = mpl.colors.Normalize(vmin=rel_energy_min, vmax=rel_energy_max)
            cbar_label = 'Multi-objective error score'
            cref = rel_energy_history
        elif energy_color == 'absolute':
            if energy_scale == 'linear':
                norm = mpl.colors.Normalize(vmin=abs_energy_min, vmax=abs_energy_max)
                cbar_label = 'Total objective error'
            elif energy_scale == 'log':
                if abs_energy_min > 0.:
                    norm = mpl.colors.LogNorm(vmin=abs_energy_min, vmax=abs_energy_max)
                    cbar_label = 'Total objective error (log scale)'
                else:
                    norm = mpl.colors.Normalize(vmin=abs_energy_min, vmax=abs_energy_max)
                    cbar_label = 'Total objective error'
            else:
                raise RuntimeError('OptimizationHistory.plot: energy_scale must be either \'linear\' or \'log\'')
            cref = abs_energy_history
        else:
            raise RuntimeError('OptimizationHistory.plot: energy_color must be either \'relative\' or \'absolute\'')

        if 'parameters' in categories:
            for param_name in categories['parameters']:
                param_min, param_max, param_med = get_group_stats(param_history[param_name])

                fig, axes = plt.subplots(1, figsize=(7., 4.8))
                for i in range(max_iter):
                    axes.scatter(np.ones(len(param_history[param_name]['population'][i])) * (i + 1),
                                 param_history[param_name]['population'][i], c=cref['population'][i],
                                 cmap=cmap, norm=norm, linewidth=0., alpha=0.2, s=5.)
                    if show_failed:
                        axes.scatter(np.ones(len(param_history[param_name]['failed'][i])) * (i + 1),
                                     param_history[param_name]['failed'][i], c='grey', linewidth=0, alpha=0.2,
                                     s=5.)
                    axes.scatter(np.ones(len(param_history[param_name]['specialists'][i])) * (i + 1),
                                 param_history[param_name]['specialists'][i],
                                 c=cref['specialists'][i],
                                 cmap=cmap, norm=norm, linewidth=0., alpha=0.2, s=5.)
                    axes.scatter(np.ones(len(param_history[param_name]['survivors'][i])) * (i + 1),
                                 param_history[param_name]['survivors'][i], c=cref['survivors'][i],
                                 cmap=cmap, norm=norm, linewidth=0., alpha=0.3, s=10.)
                axes.plot(range(1, max_iter + 1), param_med, c='r')
                axes.set_ylabel('Parameter value')
                axes.set_xlabel('Number of iterations')
                axes.set_title('Parameter: %s' % param_name)
                box = axes.get_position()
                axes.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
                legend_elements = [Line2D([0], [0], color='r', lw=2, label='Median')]
                axes.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.2),
                            fancybox=True, frameon=False, ncol=len(legend_elements))
                divider = make_axes_locatable(axes)
                cax = divider.append_axes('right', size='3%', pad=0.1)
                cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
                cbar.set_label(cbar_label, rotation=-90)
                cbar.ax.get_yaxis().labelpad = 15
                clean_axes(axes)
                fig.show()

        if 'features' in categories:
            for feature_name in categories['features']:
                feature_min, feature_max, feature_med = get_group_stats(feature_history[feature_name])

                fig, axes = plt.subplots(1, figsize=(7., 4.8))
                for i in range(max_iter):
                    axes.scatter(np.ones(len(feature_history[feature_name]['population'][i])) * (i + 1),
                                 feature_history[feature_name]['population'][i], c=cref['population'][i],
                                 cmap=cmap, norm=norm, linewidth=0., alpha=0.2, s=5.)
                    axes.scatter(np.ones(len(feature_history[feature_name]['specialists'][i])) * (i + 1),
                                 feature_history[feature_name]['specialists'][i],
                                 c=cref['specialists'][i],
                                 cmap=cmap, norm=norm, linewidth=0., alpha=0.2, s=5.)
                    axes.scatter(np.ones(len(feature_history[feature_name]['survivors'][i])) * (i + 1),
                                 feature_history[feature_name]['survivors'][i], c=cref['survivors'][i],
                                 cmap=cmap, norm=norm, linewidth=0., alpha=0.3, s=10.)

                axes.plot(range(1, max_iter + 1), feature_med, c='r')
                axes.set_xlabel('Number of iterations')
                axes.set_ylabel('Feature value')
                axes.set_title('Feature: %s' % feature_name)
                box = axes.get_position()
                axes.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
                legend_elements = [Line2D([0], [0], color='r', lw=2, label='Median')]
                axes.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.2),
                            fancybox=True, frameon=False, ncol=len(legend_elements))
                divider = make_axes_locatable(axes)
                cax = divider.append_axes('right', size='3%', pad=0.1)
                cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
                cbar.set_label(cbar_label, rotation=-90)
                cbar.ax.get_yaxis().labelpad = 15
                clean_axes(axes)
                fig.show()

        if 'objectives' in categories:
            for objective_name in categories['objectives']:
                objective_min, objective_max, objective_med = get_group_stats(objective_history[objective_name])

                fig, axes = plt.subplots(1, figsize=(7., 4.8))
                for i in range(max_iter):
                    axes.scatter(np.ones(len(objective_history[objective_name]['population'][i])) * (i + 1),
                                 objective_history[objective_name]['population'][i], c=cref['population'][i],
                                 cmap=cmap, norm=norm, linewidth=0., alpha=0.2, s=5.)
                    axes.scatter(np.ones(len(objective_history[objective_name]['specialists'][i])) * (i + 1),
                                 objective_history[objective_name]['specialists'][i],
                                 c=cref['specialists'][i],
                                 cmap=cmap, norm=norm, linewidth=0., alpha=0.2, s=5.)
                    axes.scatter(np.ones(len(objective_history[objective_name]['survivors'][i])) * (i + 1),
                                 objective_history[objective_name]['survivors'][i], c=cref['survivors'][i],
                                 cmap=cmap, norm=norm, linewidth=0., alpha=0.3, s=10.)
                if energy_scale == 'log':
                    if objective_min > 0.:
                        axes.semilogy(range(1, max_iter + 1), objective_med, c='r')
                        axes.set_ylabel('Objective error (log scale)')
                    else:
                        axes.plot(range(1, max_iter + 1), objective_med, c='r')
                        axes.set_ylabel('Objective error')
                elif energy_scale == 'linear':
                    axes.plot(range(1, max_iter + 1), objective_med, c='r')
                    axes.set_ylabel('Objective error')
                else:
                    raise RuntimeError('OptimizationHistory.plot: energy_scale must be either \'linear\' or \'log\'')
                axes.plot(range(1, max_iter + 1), objective_med, c='r')
                axes.set_xlabel('Number of iterations')
                axes.set_title('Objective: %s' % objective_name)
                box = axes.get_position()
                axes.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
                legend_elements = [Line2D([0], [0], color='r', lw=2, label='Median')]
                axes.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.2),
                            fancybox=True, frameon=False, ncol=len(legend_elements))
                divider = make_axes_locatable(axes)
                cax = divider.append_axes('right', size='3%', pad=0.1)
                cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
                cbar.set_label(cbar_label, rotation=-90)
                cbar.ax.get_yaxis().labelpad = 15
                clean_axes(axes)
                fig.show()

    def _onpick(self, event, annot, fig, ax, sc, x_name, y_name, z_name,
                this_x_arr, this_y_arr, this_z_arr, num_models):
        """
        for dumb_plot
        adapted from stackoverflow answer
        """
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                self._update_annot(
                    annot, sc, ind, x_name, y_name, z_name, this_x_arr, this_y_arr,
                    this_z_arr, self.total_models, num_models)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    @staticmethod
    def _update_annot(annot, sc, ind, x_name, y_name, z_name, x_arr, y_arr,
                      z_arr, total_models, num_models_to_plot):
        """for dumb_plot"""
        idx = ind["ind"][0]
        pos = sc.get_offsets()[idx]
        model_num = total_models - num_models_to_plot + idx
        annot.xy = pos
        text = "Model number %s" % model_num
        print(text)
        print("    %s = %s" % (x_name, x_arr[model_num]))
        print("    %s = %s" % (y_name, y_arr[model_num]))
        print("    %s = %s" % (z_name, z_arr[model_num]))
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor('white')
        annot.get_bbox_patch().set_alpha(0.4)

    def _check_name(self, var_name):
        category = None
        found = False
        if var_name in self.param_names:
            found = True
            category = 'parameters'
        if var_name in self.feature_names:
            if found:
                raise RuntimeError("The variable name %s is ambiguous. It could refer "
                                   "to %s or %s." % (var_name, category, 'features'))
            found = True
            category = 'features'
        if var_name in self.objective_names:
            if found:
                raise RuntimeError("The variable name %s is ambiguous. It could refer "
                                   "to %s or %s." % (var_name, category, 'objectives'))
            found = True
            category = 'objectives'
        if not found:
            raise RuntimeError("%s is not a valid variable name.\n"
                               "Parameter names: %s\n"
                               "Feature names: %s.\n"
                               "Objective names: %s.\n"
                               % (var_name, self.param_names, self.feature_names, self.objective_names)
                               )
        return category

    def _get_idx_from_name(self, var_name, category):
        var_idx = None
        if category[0] == 'p':
            if var_name in self.param_names:
                var_idx = self.param_names.index(var_name)
        elif category[0] == 'o':
            if var_name in self.objective_names:
                var_idx = self.objective_names.index(var_name)
        elif category[0] == 'f':
            if var_name in self.feature_names:
                var_idx = self.feature_names.index(var_name)
        if var_idx is None:
            raise RuntimeError("You specified %s for %s, but the variable name can't "
                               "be found for that category." % (category, var_name))
        return var_idx

    def _name_to_idx_and_cat(self, var_name, category=None):
        """
        two paths: one for unique variable name and one for duplicated
        (category user-specified) (e.g. features and objectives may have 
        some overlapped names)
        """
        if category is None:
            category = self._check_name(var_name)
        idx = self._get_idx_from_name(var_name, category.lower())
        return idx, category

    def _convert_val_to_matrix(self):
        """allows slicing for easier plotting"""
        from nested.lsa import pop_to_matrix
        self.param_matrix, self.feat_matrix = pop_to_matrix(self, 'p', 'f', ['p'], ['o'])
        _, self.obj_matrix = pop_to_matrix(self, 'p', 'o', ['p'], ['o'])

    def _get_var_col(self, idx, cat):
        if cat[0] == 'p':
            return self.param_matrix[:, idx]
        elif cat[0] == 'f':
            return self.feat_matrix[:, idx]
        elif cat[0] == 'o':
            return self.obj_matrix[:, idx]

    def _get_best_values(self, x_idx, y_idx, z_idx, x_category, y_category, z_category):
        """values for best model"""
        def get_single_val(history, idx, cat):
            if cat is None: return
            if cat[0] == 'p':
                return history.best_model.x[idx]
            elif cat[0] == 'f':
                return history.best_model.features[idx]
            elif cat[0] == 'o':
                return history.best_model.objectives[idx]
        x_val = get_single_val(self, x_idx, x_category)
        y_val = get_single_val(self, y_idx, y_category)
        z_val = get_single_val(self, z_idx, z_category)
        return x_val, y_val, z_val

    def dumb_plot(self, x_axis, y_axis, z_axis="Summed objectives", x_category=None,
                  y_category=None, z_category=None, alpha=1., num_models=None, last_third=False):
        """
        plots any two variables against each other. does not use the filtered set of points gathered during
        sensitivity analysis.

        :param x_axis: string. name of in/dependent variable
        :param y_axis: string
        :param z_axis: string
        :param x_category: string, e.g. 'parameters', 'features', 'objectives'
        :param y_category: string
        :param z_category: string
        :param alpha: float between 0 and 1; transparency of scatter points
        :param num_models: int or None. if None, plot all models. else, plot the last num_models.
        :param last_third: bool. if True, use only the values associated with the last third of the optimization
        """
        import matplotlib.pyplot as plt
        if self.param_matrix is None:
            self._convert_val_to_matrix()

        x_idx, x_category = self._name_to_idx_and_cat(x_axis, x_category)
        x_arr = self._get_var_col(x_idx, x_category)
        y_idx, y_category = self._name_to_idx_and_cat(y_axis, y_category)
        y_arr = self._get_var_col(y_idx, y_category)
        if z_axis != "Summed objectives":
            z_idx, z_category = self._name_to_idx_and_cat(z_axis, z_category)
            z_arr = self._get_var_col(z_idx, z_category)
        else:
            z_idx = None
            z_arr = self.summed_obj

        fig, ax = plt.subplots()
        if num_models is not None:
            num_models = min(int(num_models), self.total_models)
        elif last_third:
            num_models = self.total_models // 3
        else:
            num_models = self.total_models
        sc = plt.scatter(x_arr[-num_models:] , y_arr[-num_models:],
                         c=z_arr[-num_models:], cmap='viridis_r', alpha=alpha)
        if num_models != self.total_models:
            plt.title("Last {} models".format(num_models))
        else:
            plt.title("All models")

        plt.colorbar().set_label(z_axis)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        if self.best_model is not None:
            x_best, y_best, z_best = self._get_best_values(
                x_idx, y_idx, z_idx, x_category, y_category, z_category)
            plt.scatter(x_best, y_best, color='red', marker='+')
            print("Best model")
            print("    %s = %s" % (x_axis, x_best))
            print("    %s = %s" % (y_axis, y_best))
            if z_best is None:
                z_best = sum(self.best_model.objectives)
            print("    %s = %s" % (z_axis, z_best))

        annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)
        fig.canvas.mpl_connect('button_press_event',
                               lambda event: self._onpick(
                                   event, annot, fig, ax, sc, x_axis, y_axis, z_axis,
                                   x_arr, y_arr, z_arr, num_models)
                               )
        plt.show()

    def jupyter_plot(self):
        import matplotlib.pyplot as plt
        from ipywidgets import widgets
        categories = ['parameters', 'features', 'objectives']

        if self.param_matrix is None:
            from nested.lsa import pop_to_matrix
            self.param_matrix, self.feat_matrix = pop_to_matrix(self, 'p', 'f', ['p'], ['o'])
            _, self.obj_matrix = pop_to_matrix(self, 'p', 'o', ['p'], ['o'])

        def on_X_change(_):
            if X_category.value == 'parameters':
                inp.options = self.param_names
                inp.value = self.param_names[0]
            elif X_category.value == 'features':
                inp.options = self.feature_names
                inp.value = self.feature_names[0]
            else:
                inp.options = self.objective_names
                inp.value = self.objective_names[0]

        def on_y_change(_):
            if y_category.value == 'parameters':
                out.options = self.param_names
                out.value = self.param_names[0]
            elif y_category.value == 'features':
                out.options = self.feature_names
                out.value = self.feature_names[0]
            else:
                out.options = self.objective_names
                out.value = self.objective_names[0]

        def plot_widget(X_cat, y_cat, this_inp, this_out):
            x = get_column(X_cat, this_inp)
            y = get_column(y_cat, this_out)
            plt.figure(figsize=(10, 8))
            if (type(x) is int and x == -1) or (type(y) is int and y == -1):  # error
                plt.scatter([], [])
                plt.title("Key error")
            else:
                plt.scatter(x, y, c=self.summed_obj, cmap='viridis_r')
                plt.title("All models")

            plt.colorbar().set_label("Summed objectives")
            plt.xlabel(this_inp)
            plt.ylabel(this_out)
            plt.show()

        def get_column(category, member_name):
            """str -> vec"""
            idx = self._get_idx_from_name(member_name, category)
            vec = self._get_var_col(idx, category)
            return vec

        X_category = widgets.Dropdown(
            options=categories,
            value=categories[0],
            description="IV category",
            disabled=False,
        )

        y_category = widgets.Dropdown(
            options=categories,
            value=categories[2],
            description="DV category",
            disabled=False,
        )

        X_category.observe(on_X_change)
        y_category.observe(on_y_change)

        inp = widgets.Dropdown(
            options=self.param_names,
            value=self.param_names[0],
            description="IV",
            disabled=False,
        )

        out = widgets.Dropdown(
            options=self.objective_names,
            value=self.objective_names[0],
            description="DV",
            disabled=False,
        )
        widgets.interact(
            plot_widget, X_cat=X_category, y_cat=y_category, this_inp=inp, this_out=out)

    def get_model_from_number(self, num):
        flat_li = [model for sub in self.generations for model in sub]
        return flat_li[num]

    def save(self, file_path, n=None):
        """
        Adds data from the most recent n generations to the hdf5 file.
        :param file_path: str
        :param n: str or int
        """
        start_time = time.time()
        io = 'w' if n == 'all' else 'a'
        with h5py.File(file_path, io) as f:
            if 'param_names' not in f.attrs:
                set_h5py_attr(f.attrs, 'param_names', self.param_names)
            if 'feature_names' not in f.attrs and self.feature_names is not None:
                set_h5py_attr(f.attrs, 'feature_names', self.feature_names)
            if 'objective_names' not in f.attrs and self.objective_names is not None:
                set_h5py_attr(f.attrs, 'objective_names', self.objective_names)
            if 'path_length' not in f.attrs:
                f.attrs['path_length'] = self.path_length
            if 'normalize' not in f.attrs:
                set_h5py_attr(f.attrs, 'normalize', self.normalize)
            if 'user_attribute_names' not in f.attrs and len(self.attributes) > 0:
                set_h5py_attr(f.attrs, 'user_attribute_names', list(self.attributes.keys()))
            if n is None:
                n = 1
            elif n == 'all':
                n = len(self.generations)
            elif not isinstance(n, int):
                n = 1
                print('OptimizationHistory: defaulting to exporting last generation to file.')
            gen_index = len(self.generations) - n
            if gen_index < 0:
                gen_index = 0
                n = len(self.generations)
                if n != 0:
                    print('OptimizationHistory: defaulting to exporting all %i generations to file.' % n)

            # save history
            j = n
            while n > 0:
                if str(gen_index) in f:
                    print('OptimizationHistory: generation %s already exported to file.' % str(gen_index))
                else:
                    f.create_group(str(gen_index))
                    for key in self.attributes:
                        set_h5py_attr(f[str(gen_index)].attrs, key, self.attributes[key][gen_index])
                    f[str(gen_index)].attrs['count'] = self.count
                    if self.min_objectives[gen_index] is not None and len(self.min_objectives[gen_index]) > 0 and \
                            self.max_objectives[gen_index] is not None and len(self.max_objectives[gen_index]) > 0:
                        f[str(gen_index)].create_dataset(
                            'min_objectives',
                            data=[None2nan(val) for val in self.min_objectives[gen_index]],
                            compression='gzip')
                        f[str(gen_index)].create_dataset(
                            'max_objectives',
                            data=[None2nan(val) for val in self.max_objectives[gen_index]],
                            compression='gzip')
                    for group_name, population in \
                            zip(['population', 'survivors', 'specialists', 'prev_survivors', 'prev_specialists',
                                 'failed'],
                                [self.generations[gen_index], self.survivors[gen_index], self.specialists[gen_index],
                                 self.prev_survivors[gen_index], self.prev_specialists[gen_index],
                                 self.failed[gen_index]]):
                        f[str(gen_index)].create_group(group_name)
                        for i, individual in enumerate(population):
                            f[str(gen_index)][group_name].create_group(str(i))
                            f[str(gen_index)][group_name][str(i)].attrs['id'] = None2nan(individual.model_id)
                            f[str(gen_index)][group_name][str(i)].create_dataset(
                                'x', data=[None2nan(val) for val in individual.x], compression='gzip')
                            if group_name != 'failed':
                                f[str(gen_index)][group_name][str(i)].attrs['energy'] = None2nan(individual.energy)
                                f[str(gen_index)][group_name][str(i)].attrs['rank'] = None2nan(individual.rank)
                                f[str(gen_index)][group_name][str(i)].attrs['distance'] = \
                                    None2nan(individual.distance)
                                f[str(gen_index)][group_name][str(i)].attrs['fitness'] = \
                                    None2nan(individual.fitness)
                                f[str(gen_index)][group_name][str(i)].attrs['survivor'] = \
                                    None2nan(individual.survivor)
                                if individual.features is not None:
                                    f[str(gen_index)][group_name][str(i)].create_dataset(
                                        'features', data=[None2nan(val) for val in individual.features],
                                        compression='gzip')
                                if individual.objectives is not None:
                                    f[str(gen_index)][group_name][str(i)].create_dataset(
                                        'objectives', data=[None2nan(val) for val in individual.objectives],
                                        compression='gzip')
                                if individual.normalized_objectives is not None:
                                    f[str(gen_index)][group_name][str(i)].create_dataset(
                                        'normalized_objectives',
                                        data=[None2nan(val) for val in individual.normalized_objectives],
                                        compression='gzip')
                n -= 1
                gen_index += 1

        if j != 0:
            print('OptimizationHistory: saving %i generations (up to generation %i) to file: %s took %.2f s' %
                  (j, gen_index - 1, file_path, time.time() - start_time))

    def load(self, file_path):
        """

        :param file_path: str
        """
        start_time = time.time()
        if not os.path.isfile(file_path):
            raise IOError('OptimizationHistory: invalid file path: %s' % file_path)
        self.generations = []  # a list of populations, each corresponding to one generation
        self.survivors = []  # a list of populations (some may be empty)
        self.specialists = []  # a list of populations (some may be empty)
        self.prev_survivors = []  # a list of populations (some may be empty)
        self.prev_specialists = []  # a list of populations (some may be empty)
        self.failed = []  # a list of populations (some may be empty)
        self.min_objectives = []  # list of array of float
        self.max_objectives = []  # list of array of float
        self.attributes = {}  # a dict containing lists of user specified attributes
        with h5py.File(file_path, 'r') as f:
            self.param_names = list(get_h5py_attr(f.attrs, 'param_names'))
            self.feature_names = list(get_h5py_attr(f.attrs, 'feature_names'))
            self.objective_names = list(get_h5py_attr(f.attrs, 'objective_names'))
            self.path_length = int(f.attrs['path_length'])
            self.normalize = get_h5py_attr(f.attrs, 'normalize')
            if 'user_attribute_names' in f.attrs and len(f.attrs['user_attribute_names']) > 0:
                for key in get_h5py_attr(f.attrs, 'user_attribute_names'):
                    self.attributes[key] = []

            for gen_index in range(len(f)):
                for key in self.attributes:
                    if key in f[str(gen_index)].attrs:
                        self.attributes[key].append(get_h5py_attr(f[str(gen_index)].attrs, key))
                self.count = int(f[str(gen_index)].attrs['count'])
                if 'min_objectives' in f[str(gen_index)]:
                    self.min_objectives.append(f[str(gen_index)]['min_objectives'][:])
                else:
                    self.min_objectives.append([])
                if 'max_objectives' in f[str(gen_index)]:
                    self.max_objectives.append(f[str(gen_index)]['max_objectives'][:])
                else:
                    self.max_objectives.append([])
                population, survivors, specialists, prev_survivors, prev_specialists, failed = [], [], [], [], [], []
                for group_name, subpopulation in \
                        zip(['population', 'survivors', 'specialists', 'prev_survivors', 'prev_specialists', 'failed'],
                            [population, survivors, specialists, prev_survivors, prev_specialists, failed]):
                    if group_name not in f[str(gen_index)].keys():
                        continue
                    group = f[str(gen_index)][group_name]
                    for i in range(len(group)):
                        indiv_data = group[str(i)]
                        model_id = nan2None(indiv_data.attrs['id'])
                        individual = Individual(indiv_data['x'][:], model_id=model_id)
                        if group_name != 'failed':
                            if 'features' in indiv_data:
                                individual.features = indiv_data['features'][:]
                            if 'objectives' in indiv_data:
                                individual.objectives = indiv_data['objectives'][:]
                            if 'normalized_objectives' in indiv_data:
                                individual.normalized_objectives = indiv_data['normalized_objectives'][:]
                            individual.energy = nan2None(indiv_data.attrs.get('energy', np.nan))
                            individual.rank = nan2None(indiv_data.attrs.get('rank', np.nan))
                            individual.distance = nan2None(indiv_data.attrs.get('distance', np.nan))
                            individual.fitness = nan2None(indiv_data.attrs.get('fitness', np.nan))
                            individual.survivor = nan2None(indiv_data.attrs.get('survivor', np.nan))
                        subpopulation.append(individual)
                self.generations.append(population)
                self.survivors.append(survivors)
                self.specialists.append(specialists)
                self.prev_survivors.append(prev_survivors)
                self.prev_specialists.append(prev_specialists)
                self.failed.append(failed)
        print('OptimizationHistory: loading %i generations from file: %s took %.2f s' %
              (len(self.generations), file_path, time.time() - start_time))

    def global_renormalize_objectives(self):
        for i in range(len(self.generations)):
            this_population = self.generations[i] + self.survivors[i] + self.specialists[i]
            assign_normalized_objectives(this_population, min_objectives=self.min_objectives[-1],
                                         max_objectives=self.max_objectives[-1])
            assign_relative_energy(this_population)


class RelativeBoundedStep(object):
    """
    Step-taking method for use with PopulationAnnealing. Steps each parameter within specified absolute and/or relative
    bounds. Explores the range in log10 space when the range is >= 2 orders of magnitude (except if the range spans
    zero).
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
        if bounds is None:
            raise RuntimeError('RelativeBoundedStep: missing required parameter bounds.')
        if random is None:
            self.random = np.random
        else:
            self.random = random
        if param_names is None:
            raise RuntimeError('RelativeBoundedStep: missing required list of parameter names.')
        self.param_names = param_names
        self.param_indexes = {param: i for i, param in enumerate(param_names)}
        xmin = []
        xmax = []
        for bound in bounds:
            if bound[0] is None:
                raise RuntimeError('RelativeBoundedStep: missing required parameter bounds.')
            xmin.append(bound[0])
            if bound[1] is None:
                raise RuntimeError('RelativeBoundedStep: missing required parameter bounds.')
            xmax.append(bound[1])
        if not np.all(xmax >= xmin):
            raise ValueError('RelativeBoundedStep: Misspecified bounds: not all xmin <= xmax.')
        self.xmin = np.array(xmin)
        self.xmax = np.array(xmax)
        self.x_range = np.subtract(self.xmax, self.xmin)
        if x0 is None:
            print('RelativeBoundedStep: starting parameters not specified; choosing random initial parameters.')
            x0 = [self.random.uniform(self.xmin[i], self.xmax[i]) for i in range(len(bounds))]
        else:
            for i in range(len(x0)):
                if x0[i] is None:
                    print('RelativeBoundedStep: starting value for parameter: %s not specified; choosing random '
                          'value.' % self.param_names[i])
                    x0[i] = self.random.uniform(self.xmin[i], self.xmax[i])
        self.x0 = np.array(x0)
        self.abs_order_mag = []
        for i in range(len(xmin)):
            xi_logmin, xi_logmax, offset, factor = logmod_bounds(xmin[i], xmax[i])
            self.abs_order_mag.append(xi_logmax - xi_logmin)
        self.rel_bounds = rel_bounds
        if not self.check_bounds(self.x0):
            raise ValueError('RelativeBoundedStep: Starting parameters are not within specified bounds.')

    def __call__(self, current_x=None, stepsize=None, wrap=None):
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
        if current_x is None:
            current_x = self.x0
        x = np.array(current_x)
        for i in range(len(x)):
            new_xi = self.generate_param(x[i], i, self.xmin[i], self.xmax[i], stepsize, wrap, self.disp)
            x[i] = new_xi
        if self.rel_bounds is not None:
            x = self.apply_rel_bounds(x, stepsize, self.rel_bounds, self.disp)
        return x

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
            xi_logmin, xi_logmax, offset, factor = logmod_bounds(xi_min, xi_max)
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
        xi_log = logmod(xi, offset, factor)
        step = stepsize * self.abs_order_mag[i] / 2.
        if disp:
            print('Before: log_xi: %.4f, step: %.4f, xi_logmin: %.4f, xi_logmax: %.4f' % (xi_log, step, xi_logmin,
                                                                                          xi_logmax))
        if wrap:
            step = min(step, xi_logmax - xi_logmin)
            delta = self.random.uniform(-step, step)
            step_xi_log = xi_log + delta
            if xi_logmin > step_xi_log:
                step_xi_log = max(xi_logmax - (xi_logmin - step_xi_log), xi_logmin)
            elif xi_logmax < step_xi_log:
                step_xi_log = min(xi_logmin + (step_xi_log - xi_logmax), xi_logmax)
            new_xi = logmod_inv(step_xi_log, offset, factor)
        else:
            step_xi_logmin = max(xi_logmin, xi_log - step)
            step_xi_logmax = min(xi_logmax, xi_log + step)
            new_xi_log = self.random.uniform(step_xi_logmin, step_xi_logmax)
            new_xi = logmod_inv(new_xi_log, offset, factor)
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

    def __init__(self, param_names=None, feature_names=None, objective_names=None, pop_size=1, x0=None, bounds=None,
                 rel_bounds=None, wrap_bounds=False, take_step=None, evaluate=None, select=None, opt_rand_seed=None,
                 normalize='global', max_iter=50, path_length=3, initial_step_size=0.5, adaptive_step_factor=0.9,
                 survival_rate=0.2, diversity_rate=0.05, fitness_range=2, disp=False, hot_start=False,
                 history_file_path=None, specialists_survive=True, **kwargs):
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
        :param evaluate: str or callable
        :param select: str or callable
        :param opt_rand_seed: int or :class:'np.random.RandomState'
        :param normalize: str; 'global': normalize over entire history, 'local': normalize per iteration
        :param max_iter: int
        :param path_length: int
        :param initial_step_size: float in range(0., 1.]
        :param adaptive_step_factor: float in range(0., 1.]
        :param survival_rate: float in range(0., 1.]
        :param diversity_rate: float in range(0., 1.]
        :param fitness_range: int; promote additional individuals with fitness values in fitness_range
        :param disp: bool
        :param hot_start: bool
        :param history_file_path: str (path)
        :param specialists_survive: bool; whether to include specialists as survivors
        :param kwargs: dict of additional options, catches generator-specific options that do not apply
        """
        if x0 is None:
            self.x0 = None
        else:
            self.x0 = np.array(x0)
        if evaluate is None:
            self.evaluate = evaluate_population_annealing
        elif callable(evaluate):
            self.evaluate = evaluate
        elif isinstance(evaluate, str) and evaluate in globals() and callable(globals()[evaluate]):
            self.evaluate = globals()[evaluate]
        else:
            raise TypeError("PopulationAnnealing: evaluate must be callable.")
        if select is None:
            self.select = select_survivors_by_rank_and_fitness  # select_survivors_by_rank
        elif callable(select):
            self.select = select
        elif isinstance(select, str) and select in globals() and \
                callable(globals()[select]):
            self.select = globals()[select]
        else:
            raise TypeError("PopulationAnnealing: select must be callable.")
        if isinstance(opt_rand_seed, (str, bytes)):
            opt_rand_seed = int(opt_rand_seed)
        elif opt_rand_seed is None:
            opt_rand_seed = np.random.randint(4294967295)
        self.random = check_random_state(opt_rand_seed)
        self.xmin = np.array([bound[0] for bound in bounds])
        self.xmax = np.array([bound[1] for bound in bounds])
        self.history_file_path = history_file_path
        if history_file_path is not None:
            with h5py.File(history_file_path, 'a') as f:
                set_h5py_attr(f.attrs, 'opt_rand_seed', opt_rand_seed)
        self.prev_survivors = []
        self.prev_specialists = []
        max_iter = int(max_iter)
        path_length = int(path_length)
        initial_step_size = float(initial_step_size)
        adaptive_step_factor = float(adaptive_step_factor)
        survival_rate = float(survival_rate)
        diversity_rate = float(diversity_rate)
        fitness_range = int(fitness_range)
        if hot_start:
            if self.history_file_path is None or not os.path.isfile(self.history_file_path):
                raise IOError('PopulationAnnealing: invalid file path. Cannot hot start from stored history: %s' %
                              hot_start)
            self.history = OptimizationHistory(file_path=self.history_file_path)
            param_names = self.history.param_names
            self.path_length = self.history.path_length
            if 'step_size' in self.history.attributes:
                current_step_size = self.history.attributes['step_size'][-1]
            else:
                current_step_size = None
            if current_step_size is not None:
                initial_step_size = float(current_step_size)
            self.num_gen = len(self.history.generations)
            self.population = self.history.generations[-1]
            self.survivors = self.history.survivors[-1]
            self.specialists = self.history.specialists[-1]
            self.min_objectives = self.history.min_objectives[-1]
            self.max_objectives = self.history.max_objectives[-1]
            self.count = self.history.count
            self.normalize = self.history.normalize
            self.objectives_stored = True
        else:
            if normalize in ['local', 'global']:
                self.normalize = normalize
            else:
                raise ValueError('PopulationAnnealing: normalize argument must be either \'global\' or \'local\'')
            self.history = OptimizationHistory(param_names=param_names, feature_names=feature_names,
                                             objective_names=objective_names, path_length=path_length,
                                             normalize=self.normalize)
            self.path_length = path_length
            self.num_gen = 0
            self.population = []
            self.survivors = []
            self.specialists = []
            self.min_objectives = []
            self.max_objectives = []
            self.count = 0
            self.objectives_stored = False
        self.pop_size = int(pop_size)
        if take_step is None:
            self.take_step = RelativeBoundedStep(self.x0, param_names=param_names, bounds=bounds, rel_bounds=rel_bounds,
                                                 stepsize=initial_step_size, wrap=wrap_bounds, random=self.random)
        elif callable(take_step):
            self.take_step = take_step(self.x0, param_names=param_names, bounds=bounds,
                                       rel_bounds=rel_bounds, stepsize=initial_step_size,
                                       wrap=wrap_bounds, random=self.random)
        elif isinstance(take_step, str) and take_step in globals() and callable(globals()[take_step]):
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
        self.num_diversity_survivors = int(self.pop_size * diversity_rate)
        self.fitness_range = int(fitness_range)
        self.disp = disp
        self.specialists_survive = specialists_survive
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
                print('PopulationAnnealing: Gen %i, yielding parameters for population size %i' %
                      (self.num_gen, len(self.population)))
            self.local_time = time.time()
            sys.stdout.flush()
            generation, model_ids = [], []
            for individual in self.population:
                generation.append(individual.x)
                model_ids.append(individual.model_id)
            yield generation, model_ids
            self.num_gen += 1
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
        failed = []
        for i, objective_dict in enumerate(objectives):
            feature_dict = features[i]
            if not isinstance(objective_dict, dict):
                raise TypeError('PopulationAnnealing.update_population: objectives must be a list of dict')
            if not isinstance(feature_dict, dict):
                raise TypeError('PopulationAnnealing.update_population: features must be a list of dict')
            # empty feature_dict or objective_dict indicates a failed model
            # the first model that does not fail will be used to define the length and order of features and objectives
            # as arrays
            if self.history.feature_names is None and feature_dict:
                self.history.feature_names = sorted(list(feature_dict.keys()))
            if self.history.objective_names is None and objective_dict:
                self.history.objective_names = sorted(list(objective_dict.keys()))
            if not feature_dict or not objective_dict:
                failed.append(self.population[i])
            else:
                this_objectives = np.array([objective_dict[key] for key in self.history.objective_names])
                self.population[i].objectives = this_objectives
                this_features = np.array([feature_dict[key] for key in self.history.feature_names])
                self.population[i].features = this_features
                filtered_population.append(self.population[i])
        self.population = filtered_population
        self.history.append(self.population, prev_survivors=self.prev_survivors,
                            prev_specialists=self.prev_specialists, failed=failed,
                            step_size=self.take_step.stepsize)
        self.prev_survivors = []
        self.prev_specialists = []
        self.objectives_stored = True
        if self.disp:
            print('PopulationAnnealing: Gen %i, computing features for population size %i took %.2f s; %i individuals '
                  'failed' % (self.num_gen, len(self.population), time.time() - self.local_time, len(failed)))
        self.local_time = time.time()

        if (self.num_gen + 1) % self.path_length == 0:
            candidates = self.get_candidates()
            if len(candidates) > 0:
                self.min_objectives, self.max_objectives = \
                    get_objectives_edges(candidates, min_objectives=self.min_objectives,
                                         max_objectives=self.max_objectives, normalize=self.normalize)
                self.evaluate(candidates, min_objectives=self.min_objectives, max_objectives=self.max_objectives)
                self.specialists = get_specialists(candidates)
                self.survivors = \
                    self.select(candidates, self.num_survivors, self.num_diversity_survivors,
                                fitness_range=self.fitness_range, disp=self.disp)
                if self.disp:
                    print('PopulationAnnealing: Gen %i, evaluating iteration took %.2f s' %
                          (self.num_gen, time.time() - self.local_time))
                self.local_time = time.time()
                for individual in self.survivors:
                    individual.survivor = True
                if self.specialists_survive:
                    for individual in self.specialists:
                        individual.survivor = True
                self.history.survivors[-1] = deepcopy(self.survivors)
                self.history.specialists[-1] = deepcopy(self.specialists)
                self.history.min_objectives[-1] = deepcopy(self.min_objectives)
                self.history.max_objectives[-1] = deepcopy(self.max_objectives)
            if self.history_file_path is not None:
                self.history.save(self.history_file_path, n=self.path_length)
        sys.stdout.flush()

    def get_candidates(self):
        """
        :return: list of :class:'Individual'
        """
        candidates = []
        candidates.extend(self.history.prev_survivors[-self.path_length])
        if self.specialists_survive:
            candidates.extend(self.history.prev_specialists[-self.path_length])
        # remove duplicates
        unique_model_ids = set()
        unique_candidates = []
        for indiv in candidates:
            if indiv.model_id not in unique_model_ids:
                unique_model_ids.add(indiv.model_id)
                unique_candidates.append(indiv)
        for i in range(1, self.path_length + 1):
            unique_candidates.extend(self.history.generations[-i])
        return unique_candidates

    def init_population(self):
        """
        """
        pop_size = self.pop_size
        self.population = []
        if self.x0 is not None and self.num_gen == 0:
            self.population.append(Individual(self.x0, model_id=self.count))
            pop_size -= 1
            self.count += 1
        for i in range(pop_size):
            self.population.append(Individual(self.take_step(self.x0, stepsize=1., wrap=True), model_id=self.count))
            self.count += 1

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
            self.prev_survivors = deepcopy(self.survivors)
            self.prev_specialists = deepcopy(self.specialists)
            group = list(self.prev_survivors)
            if self.specialists_survive:
                group.extend(self.prev_specialists)
            group_size = len(group)
            for individual in group:
                individual.survivor = False
            for i in range(self.pop_size):
                individual = Individual(self.take_step(group[i % group_size].x), model_id=self.count)
                new_population.append(individual)
                self.count += 1
            self.population = new_population
        self.survivors = []
        self.specialists = []

    def step_population(self):
        """
        """
        this_pop_size = len(self.population)
        if this_pop_size == 0:
            self.init_population()
        else:
            new_population = []
            for i in range(self.pop_size):
                individual = Individual(self.take_step(self.population[i % this_pop_size].x), model_id=self.count)
                new_population.append(individual)
                self.count += 1
            self.population = new_population


class Pregenerated(object):
    def __init__(self, param_names=None, feature_names=None, objective_names=None, hot_start=False,
                 history_file_path=None, config_file_path=None, pregen_param_file_path=None, evaluate=None, select=None,
                 disp=False, pop_size=50, fitness_range=2, survival_rate=.2, normalize='global',
                 specialists_survive=True, **kwargs):
        """

        :param param_names: list of str
        :param feature_names: list of str
        :param objective_names: list of str
        :param hot_start: bool, whether history_file_path is already at least partially
             evaluated
        :param history_file_path: str
        :param param_file_path: str, path to .hdf5 file of only params
        :param evaluate: func
        :param select: func
        :param disp: bool
        :param pop_size: int
        :param fitness_range: int
        :param survival_rate: float between 0 and 1
        :param normalize: str, 'local' or 'global'
        :param specialists_survive: bool
        :param kwargs:
        """
        if pregen_param_file_path is None:
            raise RuntimeError("Path to file containing parameters must be specified.")
        if evaluate is None:
            self.evaluate = evaluate_population_annealing
        elif callable(evaluate):
            self.evaluate = evaluate
        elif isinstance(evaluate, str) and evaluate in globals() and callable(globals()[evaluate]):
            self.evaluate = globals()[evaluate]
        else:
            raise TypeError("Pregenerated: evaluate must be callable.")
        if select is None:
            self.select = select_survivors_by_rank_and_fitness  # select_survivors_by_rank
        elif callable(select):
            self.select = select
        elif isinstance(select, str) and select in globals() and callable(globals()[select]):
            self.select = globals()[select]
        else:
            raise TypeError("Pregenerated: select must be callable.")
        if config_file_path is None:
            raise RuntimeError("Pregenerated: Config file path must be specified.")

        self.param_names = param_names
        self.feature_names = feature_names
        self.objective_names = objective_names
        self.disp = disp
        self.specialists_survive = specialists_survive
        self.fitness_range = int(fitness_range)

        self.hot_start = hot_start
        self.pregen_params = load_pregen(pregen_param_file_path)
        self.num_points = self.pregen_params.shape[0]
        self.pregen_param_file_path = pregen_param_file_path
        self.history_file_path = history_file_path
        self.config_file_path = config_file_path

        if hot_start and os.path.isfile(history_file_path):
            self.history = OptimizationHistory(file_path=history_file_path)
            self.population = self.history.generations[-1]
            self.survivors = self.history.survivors[-1]
            self.specialists = self.history.specialists[-1]
            self.min_objectives = self.history.min_objectives[-1]
            self.max_objectives = self.history.max_objectives[-1]
            if self.history.normalize != normalize:
                warnings.warn("Pregenerated: %s normalization was specified, but the one in the "
                              "OptimizationHistory object is %s. Defaulting to the one in history."
                              %(normalize, self.history.normalize), Warning)
            self.normalize = self.history.normalize
            self.objectives_stored = True
            self.pop_size = len(self.history.generations[0]) + len(self.history.failed[0])
            self.curr_iter = len(self.history.generations)
        else:
            self.history = OptimizationHistory(param_names=param_names, feature_names=feature_names,
                                             objective_names=objective_names, normalize=normalize, path_length=1)
            self.history.count = 0
            self.population = []
            self.survivors = []
            self.specialists = []
            self.min_objectives = []
            self.max_objectives = []
            self.objectives_stored = False
            self.pop_size = int(pop_size) # save every pop_size
            if normalize in ['local', 'global']:
                self.normalize = normalize
            else:
                raise ValueError('Pregenerated: normalize argument must be either \'global\' or \'local\'')
            self.curr_iter = 0

        if self.corruption():
            self.curr_iter -= 1
        self.start_iter = self.curr_iter
        self.max_iter = self.get_max_iter()
        survival_rate = float(survival_rate)
        self.num_survivors = max(1, int(self.pop_size * survival_rate))

        self.prev_survivors = []
        self.prev_specialists = []
        self.local_time = time.time()

    def __call__(self):
        for i in range(self.start_iter, self.max_iter):
            self.curr_iter = i
            self.curr_gid_range = range(i * self.pop_size, min((i + 1) * self.pop_size, self.num_points))
            self.population = [Individual(x=self.pregen_params[j], model_id=j) for j in self.curr_gid_range]
            self.prev_survivors = deepcopy(self.survivors)
            self.prev_specialists = deepcopy(self.specialists)
            yield [individual.x for individual in self.population], \
                  list(self.curr_gid_range)

    def update_population(self, features, objectives):
        filtered_population = []
        failed = []
        for i, objective_dict in enumerate(objectives):
            feature_dict = features[i]
            if not isinstance(objective_dict, dict):
                raise TypeError('Pregenerated.update_population: objectives must be a list of dict')
            if not isinstance(feature_dict, dict):
                raise TypeError('Pregenerated.update_population: features must be a list of dict')
            if not (all(key in objective_dict for key in self.history.objective_names) and
                    all(key in feature_dict for key in self.history.feature_names)):
                failed.append(self.population[i])
            else:
                this_objectives = np.array([objective_dict[key] for key in self.history.objective_names])
                self.population[i].objectives = this_objectives
                this_features = np.array([feature_dict[key] for key in self.history.feature_names])
                self.population[i].features = this_features
                filtered_population.append(self.population[i])
        self.population = filtered_population
        self.history.append(self.population, prev_survivors=self.prev_survivors,
                            prev_specialists=self.prev_specialists, failed=failed)
        self.prev_survivors = []
        self.prev_specialists = []
        self.objectives_stored = True
        if self.disp:
            print('Pregenerated: Iter %i, computing features for population size %i took %.2f s; %i individuals '
                  'failed' % (self.curr_iter, self.pop_size, time.time() - self.local_time, len(failed)))
        self.local_time = time.time()

        candidates = self.get_candidates()
        if len(candidates) > 0:
            self.min_objectives, self.max_objectives = \
                get_objectives_edges(candidates, min_objectives=self.min_objectives,
                                     max_objectives=self.max_objectives, normalize=self.normalize)
            self.evaluate(candidates, min_objectives=self.min_objectives, max_objectives=self.max_objectives)
            self.specialists = get_specialists(candidates)
            self.survivors = \
                self.select(candidates, self.num_survivors, fitness_range=self.fitness_range, disp=self.disp)
            if self.disp:
                print('Pregenerated: Iter %i, evaluating iteration took %.2f s' %
                      (self.curr_iter, time.time() - self.local_time))
            self.local_time = time.time()
            for individual in self.survivors:
                individual.survivor = True
            if self.specialists_survive:
                for individual in self.specialists:
                    individual.survivor = True
            self.history.survivors[-1] = deepcopy(self.survivors)
            self.history.specialists[-1] = deepcopy(self.specialists)
            self.history.min_objectives[-1] = deepcopy(self.min_objectives)
            self.history.max_objectives[-1] = deepcopy(self.max_objectives)
        if self.history_file_path is not None:
            self.history.save(self.history_file_path)
        sys.stdout.flush()

    def get_candidates(self):
        """
        TODO: remove duplicates by tracking the model_id.
        :return: list of :class:'Individual'
        """
        candidates = []
        candidates.extend(self.history.prev_survivors[-1])
        if self.specialists_survive:
            candidates.extend(self.history.prev_specialists[-1])
        # remove duplicates
        unique_model_ids = set()
        unique_candidates = []
        for indiv in candidates:
            if indiv.model_id not in unique_model_ids:
                unique_model_ids.add(indiv.model_id)
                unique_candidates.append(indiv)
        unique_candidates.extend(self.history.generations[-1])

        return unique_candidates

    def corruption(self):
        # casting bc np.sum returns a float if the list is empty
        offset = int(np.sum([len(x) for x in self.history.generations]) + np.sum([len(x) for x in self.history.failed]))
        if not self.hot_start and offset != 0:
            raise RuntimeError("Pregenerated: The hot-start flag was not provided, but some models in the "
                               "history file have already been analyzed.")
        if offset > self.num_points:
            raise RuntimeError("Pregenerated: The total number of analyzed models (%i) in the history file exceeds the "
                               "number of models in the parameters-only file (%i)." % (offset, self.num_points))

        # check if the previous model was incompletely saved
        last_gen = int(offset / self.pop_size)
        if last_gen != 0 and offset % self.pop_size == 0:
            with h5py.File(self.history_file_path, "a") as f:
                if str(last_gen) in f.keys():
                    del f[str(last_gen)]
            last_gen -= 1
        corrupt = False
        if offset > 0:
            corrupt = self.handle_possible_corruption(last_gen, offset)
        return corrupt

    def handle_possible_corruption(self, last_gen, offset):
        corrupt = False
        must_be_populated = ['population', 'survivors', 'specialists']
        default_pop_size = 50
        with h5py.File(self.history_file_path, "a") as f:
            for group_name in ['population', 'survivors', 'specialists', 'prev_survivors',
                               'prev_specialists', 'failed']:
                if group_name not in f[str(last_gen)].keys() or \
                        (group_name in must_be_populated and not len(f[str(last_gen)][group_name])):
                    corrupt = True
                    if 'population' in f[str(last_gen)]:
                        offset -= len(f[str(last_gen)]['population'].keys())
                        del self.history.generations[-1]
                        del self.history.survivors[-1]
                        del self.history.specialists[-1]
                        del self.history.min_objectives[-1]
                        del self.history.max_objectives[-1]
                        del self.history.failed[-1]
                        self.history.count = last_gen * len(self.history.generations[0])
                        if last_gen != 0:
                            self.population = self.history.generations[-1]
                            self.survivors = self.history.survivors[-1]
                            self.specialists = self.history.specialists[-1]
                            self.min_objectives = self.history.min_objectives[-1]
                            self.max_objectives = self.history.max_objectives[-1]
                        else:
                            self.population = []
                            self.survivors = []
                            self.specialists = []
                            self.min_objectives = []
                            self.max_objectives = []
                            self.objectives_stored = False
                            self.pop_size = default_pop_size
                    del f[str(last_gen)]
                    break
        return corrupt

    def get_max_iter(self):
        max_iter = int(self.num_points / self.pop_size)
        if self.num_points % self.pop_size != 0:
            max_iter += 1
        return max_iter


class Sobol(Pregenerated):
    def __init__(self, param_names=None, feature_names=None, objective_names=None, disp=False,
                 hot_start=False, pregen_param_file_path=None, history_file_path=None, num_models=None,
                 config_file_path=None, evaluate=None, select=None, survival_rate=.2, fitness_range=2, pop_size=50,
                 normalize='global', specialists_survive=True, **kwargs):
        """

        :param num_models: int, upper bound for number of models. required if hot_start is False
        """
        if not hot_start and num_models is None:
            raise RuntimeError("Sobol: num_models must be provided.")
        if pregen_param_file_path is None:
            if hot_start:
                raise RuntimeError("Sobol: hot-start flag provided, but pregen_param_file_path was not specified.")
            pregen_param_file_path = "data/%s_Sobol_sequence.hdf5" % (datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))
        elif not hot_start:
            raise RuntimeError("Sobol: hot-start flag not provided, but pregen_param_file_path was specified.")

        if not os.path.isfile(pregen_param_file_path):
            if hot_start:
                raise RuntimeError("Sobol: hot-start flag provided, but pregen_param_file_path (%s) is empty."
                                   % pregen_param_file_path)
            self.n = int(num_models) // (2 * len(param_names) + 2)
            if self.n == 0:
                raise RuntimeError(
                    "Sobol: Too low of a ceiling on the number of models (num_models). The user specified "
                    "at most %s models, but at least %s models are needed, preferably on the "
                    "order of a hundred to ten thousand times that."
                    % (num_models, 2 * len(param_names) + 2))
            self.pregen_params = generate_sobol_seq(config_file_path, self.n, pregen_param_file_path)

        super().__init__(
            param_names=param_names, feature_names=feature_names, objective_names=objective_names,
            hot_start=hot_start,
            history_file_path=history_file_path, config_file_path=config_file_path,
            pregen_param_file_path=pregen_param_file_path,
            evaluate=evaluate, select=select, disp=disp,
            pop_size=pop_size, fitness_range=fitness_range, survival_rate=survival_rate, normalize=normalize,
            specialists_survive=specialists_survive, **kwargs
        )
        if hot_start:
            self.n = self.compute_n()

        #if self.curr_iter >= self.max_iter:
        #    sobol_analysis(config_file_path, self.history)
        print("Sobol: the total number of models is %i. n is %i." % (self.num_points, self.n))
        sys.stdout.flush()

    #def update_population(self, features, objectives):
    #    """
    #    finds matching individuals in OptimizationHistory object modifies them.
    #    also modifies hdf5 file containing the PS object
    #    """
    #    Pregenerated.update_population(self, features, objectives)
    #    # after last iter, before it's incremented
    #    #if self.curr_iter >= self.max_iter - 1:
    #    #    print("Sobol: performing sensitivity analysis...")
    #    #    sys.stdout.flush()
    #    #    sobol_analysis(self.config_file_path, self.history)

    def compute_n(self):
        """ if the user already generated a Sobol sequence, n is inferred """
        return self.num_points // (2 * len(self.param_names) + 2)


class OptimizationReport(object):
    """
    Convenience object to browse optimization results.
        survivors: list of :class:'Individual',
        specialists: dict: {objective_name: :class:'Individual'},
        param_names: list of str,
        objective_names: list of str,
        feature_names: list of str
    """
    def __init__(self, history=None, file_path=None):
        """
        Can either quickly load optimization results from a file, or report from an already loaded instance of
            :class:'OptimizationHistory'.
        :param history: :class:'OptimizationHistory'
        :param file_path: str (path)
        """
        self.file_path = file_path
        self.history = history
        if self.history is not None:
            self.param_names = history.param_names
            self.feature_names = history.feature_names
            self.objective_names = history.objective_names
            self.survivors = deepcopy(history.survivors[-1])
            self.specialists = dict()
            for i, objective in enumerate(self.objective_names):
                self.specialists[objective] = history.specialists[-1][i]
        elif self.file_path is None or not os.path.isfile(self.file_path):
            raise RuntimeError('get_optimization_report: problem loading optimization history from the specified path: '
                               '%s' % self.file_path)
        else:
            with h5py.File(self.file_path, 'r') as f:
                self.param_names = get_h5py_attr(f.attrs, 'param_names')
                self.feature_names = get_h5py_attr(f.attrs, 'feature_names')
                self.objective_names = get_h5py_attr(f.attrs, 'objective_names')
                self.survivors = []
                last_gen_key = str(len(f) - 1)
                group = f[last_gen_key]['survivors']
                for i in range(len(group)):
                    indiv_data = group[str(i)]
                    model_id = nan2None(indiv_data.attrs['id'])
                    individual = Individual(indiv_data['x'][:], model_id=model_id)
                    individual.features = indiv_data['features'][:]
                    individual.objectives = indiv_data['objectives'][:]
                    individual.normalized_objectives = indiv_data['normalized_objectives'][:]
                    individual.energy = nan2None(indiv_data.attrs['energy'])
                    individual.rank = nan2None(indiv_data.attrs['rank'])
                    individual.distance = nan2None(indiv_data.attrs['distance'])
                    individual.fitness = nan2None(indiv_data.attrs['fitness'])
                    individual.survivor = nan2None(indiv_data.attrs['survivor'])
                    self.survivors.append(individual)
                self.specialists = dict()
                group = f[last_gen_key]['specialists']
                for i, objective in enumerate(self.objective_names):
                    indiv_data = group[str(i)]
                    model_id = nan2None(indiv_data.attrs['id'])
                    individual = Individual(indiv_data['x'][:], model_id=model_id)
                    individual.features = indiv_data['features'][:]
                    individual.objectives = indiv_data['objectives'][:]
                    individual.normalized_objectives = indiv_data['normalized_objectives'][:]
                    individual.energy = nan2None(indiv_data.attrs['energy'])
                    individual.rank = nan2None(indiv_data.attrs['rank'])
                    individual.distance = nan2None(indiv_data.attrs['distance'])
                    individual.fitness = nan2None(indiv_data.attrs['fitness'])
                    individual.survivor = nan2None(indiv_data.attrs['survivor'])
                    self.specialists[objective] = individual

    def report(self, indiv):
        """

        :param indiv: :class:'Individual'
        """
        print('params:')
        print_param_array_like_yaml(indiv.x, self.param_names)
        print('features:')
        print_param_array_like_yaml(indiv.features, self.feature_names)
        print('objectives:')
        print_param_array_like_yaml(indiv.objectives, self.objective_names)
        sys.stdout.flush()

    def report_best(self):
        self.report(self.survivors[0])

    def get_marder_group(self, size=5, order=1000, threshold=0.15, reference_x=None, plot=False):
        """
        Find group of models with lowest error but divergent parameters to analyze model degeneracy. Load all models
        from file. Normalize all input parameters, and compute distances from reference. If no reference is provided,
        use the 'best' model.
        :param size: int, num models to return, ordered by error
        :param order: int, num points to consider for finding local minima
        :param threshold: distance criterion to select local minima
        :param reference_x: array of float
        :param plot: bool
        :return: list of :class:'Individual'
        """
        from scipy.signal import argrelmin
        if self.history is None:
            self.history = OptimizationHistory(file_path=self.file_path)
        self.history.global_renormalize_objectives()
        population = np.array([indiv for generation in self.history.history for indiv in generation])
        param_vals = np.array([indiv.x for indiv in population])
        min_param_vals = np.min(param_vals, axis=0)
        max_param_vals = np.max(param_vals, axis=0)
        if reference_x is None:
            reference_x = self.history.survivors[-1][0].x
        else:
            min_param_vals = np.minimum(min_param_vals, reference_x)
            max_param_vals = np.maximum(max_param_vals, reference_x)

        self.min_param_vals = min_param_vals
        self.max_param_vals = max_param_vals

        normalized_param_vals = \
            [normalize_dynamic(param_vals[:, i], min_param_vals[i], max_param_vals[i]) for i in
             range(len(min_param_vals))]
        normalized_param_vals = np.array(normalized_param_vals).T
        normalized_reference_x = \
            np.array([normalize_dynamic(reference_x[i], min_param_vals[i], max_param_vals[i]) for i in
                      range(len(min_param_vals))])
        rel_energy = np.array([indiv.energy for indiv in population])
        param_distance = np.array([np.linalg.norm(normalized_param_vals[i] - normalized_reference_x)
                                   for i in range(len(normalized_param_vals))])
        sorted_indexes = np.argsort(param_distance)
        population = population[sorted_indexes]
        param_distance = param_distance[sorted_indexes]
        rel_energy = rel_energy[sorted_indexes]
        rel_min_indexes = argrelmin(rel_energy, order=order)[0]
        selected_indexes = np.where(param_distance[rel_min_indexes] > threshold)[0]
        selected_indexes = rel_min_indexes[selected_indexes]
        resorted_indexes = np.argsort(rel_energy[selected_indexes])
        selected_indexes = selected_indexes[resorted_indexes]
        selected_indexes = np.insert(selected_indexes, 0, 0)
        if plot:
            fig = plt.figure()
            plt.scatter(param_distance, rel_energy, c='lightgrey')
            plt.scatter(param_distance[selected_indexes], rel_energy[selected_indexes], c='r')
            plt.ylabel('Multi-objective error score')
            plt.xlabel('Normalized parameter distance')
            plt.title('Marder group (order=%i)' % order)
            fig.show()
        group = population[selected_indexes][:size]
        return group

    def export_params_to_yaml(self, param_file_path, population=None, labels=None):
        """
        Export params from provided population to .yaml. If labels are not provided, model_ids are used. If a
        population is not provided, by default the 'best' and specialist models are exported.
        :param param_file_path: str path
        :param population: list of :class:'Individual'
        :param labels: list of str
        """
        data = dict()
        if population is None:
            data['best'] = param_array_to_dict(self.survivors[0].x, self.param_names)
            for model_name in self.specialists:
                data[model_name] = param_array_to_dict(self.specialists[model_name].x, self.param_names)
        else:
            if labels is None:
                labels = [indiv.model_id for indiv in population]
            values = [param_array_to_dict(indiv.x, self.param_names) for indiv in population]
            data = dict(zip(labels, values))

        write_to_yaml(param_file_path, data, convert_scalars=True)


logmod = lambda x, offset, factor: np.log10(x * factor + offset)

logmod_inv = lambda logmod_x, offset, factor: ((10. ** logmod_x) - offset) / factor


def logmod_bounds(xi_min, xi_max):
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
        xi_logmin = logmod(xi_max, offset, factor)  # When the sign is flipped, the max and min will reverse
        xi_logmax = logmod(xi_min, offset, factor)
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
            xi_logmin = logmod(xi_min, offset, factor)
            xi_logmax = logmod(xi_max, offset, factor)
    else:
        offset = 0.
        factor = 1.
        xi_logmin = logmod(xi_min, offset, factor)
        xi_logmax = logmod(xi_max, offset, factor)
    return xi_logmin, xi_logmax, offset, factor


def normalize_linear(vals, min_val, max_val):
    """
    Translate and normalize values linearly.
    :param vals: array
    :param min_val: float
    :param max_val: float
    :return: array
    """
    if min_val == max_val:
        if isinstance(vals, Iterable):
            return np.zeros_like(vals)
        else:
            return 0.
    lin_range = max_val - min_val
    if isinstance(vals, Iterable):
        vals = np.subtract(vals, min_val)
        vals = np.divide(vals, lin_range)
    else:
        vals = (vals - min_val) / lin_range

    return vals


def normalize_dynamic(vals, min_val, max_val, threshold=2):
    """
    If the range of values is below the specified threshold order of magnitude, translate and normalize
    linearly. Otherwise, translate and normalize based on the distance between values in log space.
    :param vals: array
    :param min_val: float
    :param max_val: float
    :param threshold: int
    :return: array
    """
    if min_val == max_val:
        if isinstance(vals, Iterable):
            return np.zeros_like(vals)
        else:
            return 0.
    logmin, logmax, offset, factor = logmod_bounds(min_val, max_val)
    logmod_range = logmax - logmin
    if logmod_range < threshold:
        lin_range = max_val - min_val
        if isinstance(vals, Iterable):
            vals = np.subtract(vals, min_val)
            vals = np.divide(vals, lin_range)
        else:
            vals = (vals - min_val) / lin_range
    else:
        if isinstance(vals, Iterable):
            vals = [logmod(val, offset, factor) for val in vals]
            vals = np.subtract(vals, logmin)
            vals = np.divide(vals, logmod_range)
        else:
            vals = logmod(vals, offset, factor)
            vals = (vals - logmin) / logmod_range
    return vals


def get_objectives_edges(population, min_objectives=None, max_objectives=None, normalize='global'):
    """

    :param population: list of :class:'Individual'
    :param min_objectives: array
    :param max_objectives: array
    :param normalize: str; 'global': normalize over entire history, 'local': normalize per iteration
    :return: array
    """
    pop_size = len(population)
    if pop_size == 0:
        return min_objectives, max_objectives
    num_objectives = [len(individual.objectives) for individual in population if individual.objectives is not None]
    if len(num_objectives) < pop_size:
        raise RuntimeError('get_objectives_edges: objectives have not been stored for all Individuals in population')
    if normalize not in ['local', 'global']:
        raise ValueError('get_objectives_edges: normalize argument must be either \'global\' or \'local\'')
    if normalize == 'local' or min_objectives is None or len(min_objectives) == 0:
        this_min_objectives = np.array(population[0].objectives)
    else:
        this_min_objectives = np.array(min_objectives)
    if normalize == 'local' or max_objectives is None or len(max_objectives) == 0:
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
    pop_size = len(population)
    num_objectives = [len(individual.objectives) for individual in population if individual.objectives is not None]
    if len(num_objectives) < pop_size:
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


def assign_relative_energy(population):
    """
    Modifies in place the energy attribute of each Individual in the population with the sum across all normalized
    objectives.
    :param population: list of :class:'Individual'
    """
    for individual in population:
        if individual.objectives is None or individual.normalized_objectives is None or \
                len(individual.objectives) != len(individual.normalized_objectives):
            raise RuntimeError('assign_relative_energy: objectives have not been stored for all Individuals in '
                               'population')
    for individual in population:
        individual.energy = np.sum(individual.normalized_objectives)


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


def assign_normalized_objectives(population, min_objectives=None, max_objectives=None):
    """
    Modifies in place the normalized_objectives attributes of each Individual in the population
    :param population: list of :class:'Individual'
    :param min_objectives: array of float
    :param max_objectives: array of float
    """
    pop_size = len(population)
    num_objectives = [len(individual.objectives) for individual in population if individual.objectives is not None]
    if len(num_objectives) < pop_size:
        raise Exception('assign_normalized_objectives: objectives have not been stored for all Individuals in '
                        'population')
    if min_objectives is None or len(min_objectives) == 0 or max_objectives is None or len(max_objectives) == 0:
        min_objectives, max_objectives = get_objectives_edges(population)
    num_objectives = max(num_objectives)
    for individual in population:
        individual.normalized_objectives = np.zeros(num_objectives, dtype='float32')
    for m in range(num_objectives):
        if min_objectives[m] != max_objectives[m]:
            objective_vals = [individual.objectives[m] for individual in population]
            normalized_objective_vals = \
                normalize_dynamic(objective_vals, min_objectives[m], max_objectives[m])
            for val, individual in zip(normalized_objective_vals, population):
                individual.normalized_objectives[m] = val


def evaluate_population_annealing(population, min_objectives=None, max_objectives=None, disp=False, **kwargs):
    """
    Modifies in place the fitness, energy and rank attributes of each Individual in the population.
    :param population: list of :class:'Individual'
    :param min_objectives: array of float
    :param max_objectives: array of float
    :param disp: bool
    """
    if len(population) > 0:
        assign_fitness_by_dominance(population)
        assign_normalized_objectives(population, min_objectives, max_objectives)
        assign_relative_energy(population)
        assign_rank_by_fitness_and_energy(population)
    else:
        raise RuntimeError('evaluate_population_annealing: cannot evaluate empty population.')


def evaluate_random(population, disp=False, **kwargs):
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


def select_survivors_by_rank_and_fitness(population, num_survivors, num_diversity_survivors=0, fitness_range=None,
                                         disp=False, **kwargs):
    """
    Sorts the population by the rank attribute of each Individual in the population. Selects top ranked Individuals from
    each fitness group proportional to the size of each fitness group. Returns the requested number of Individuals.
    :param population: list of :class:'Individual'
    :param num_survivors: int
    :param num_diversity_survivors: int; promote additional individuals with fitness values in fitness_range
    :param fitness_range: int
    :param disp: bool
    :return: list of :class:'Individual'
    """
    fitness_vals = np.array([individual.fitness for individual in population if individual.fitness is not None])
    if len(fitness_vals) < len(population):
        raise Exception('select_survivors_by_rank_and_fitness: fitness has not been stored for all Individuals '
                        'in population')
    sorted_population = sort_by_rank(population)
    survivors = sorted_population[:num_survivors]
    remaining_population = sorted_population[len(survivors):]
    max_fitness = min(max(fitness_vals), fitness_range)
    diversity_pool = [individual for individual in remaining_population
                      if individual.fitness in range(1, max_fitness + 1)]
    diversity_pool_size = len(diversity_pool)
    if diversity_pool_size == 0:
        return survivors
    diversity_survivors = []
    fitness_groups = defaultdict(list)
    for individual in diversity_pool:
        fitness_groups[individual.fitness].append(individual)
    for fitness in range(1, max_fitness + 1):
        if len(diversity_survivors) >= num_diversity_survivors:
            break
        if len(fitness_groups[fitness]) > 0:
            this_num_survivors = \
                max(1, int(len(fitness_groups[fitness]) / diversity_pool_size * num_diversity_survivors))
            sorted_group = sort_by_rank(fitness_groups[fitness])
            diversity_survivors.extend(sorted_group[:this_num_survivors])

    return survivors + diversity_survivors[:num_diversity_survivors]


def get_specialists(population):
    """
    For each objective, find the individual in the population with the lowest objective value. Return a list of
    individuals of length number of objectives.
    :param population:
    :return: list of :class:'Individual'
    """
    pop_size = len(population)
    num_objectives = [len(individual.objectives) for individual in population if individual.objectives is not None]
    if len(num_objectives) < pop_size:
        raise RuntimeError('get_specialists: objectives have not been stored for all Individuals in population')
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
    return specialists


def load_model_params(param_names, param_file_path=None, history_file_path=None, model_keys=None, model_ids=None,
                      verbose=False):
    """

    :param param_names: list of str
    :param param_file_path: str (path)
    :param history_file_path: str (path)
    :param model_keys: list of str
    :param model_ids: list of int
    :param verbose: bool
    :return: tuple of lists
    """
    if model_keys is not None:
        requested_model_labels = list(model_keys)
    else:
        requested_model_labels = []
    if model_ids is not None:
        requested_model_ids = list(model_ids)
    else:
        requested_model_ids = []
    ids_to_labels = defaultdict(set)
    ids_to_param_arrays = dict()
    param_arrays = []
    model_labels = []
    export_keys = []
    legend = {}

    if history_file_path is not None:
        if not os.path.isfile(history_file_path):
            raise Exception('nested.analyze: invalid history_file_path: %s' % history_file_path)
        legend['source'] = history_file_path
        report = OptimizationReport(file_path=history_file_path)
        objective_names = report.objective_names
        if 'all' in requested_model_labels:
            requested_model_labels.remove('all')
            requested_model_labels.extend(['best'] + list(objective_names))
        elif len(requested_model_labels) < 1 and len(requested_model_ids) < 1:
            requested_model_labels.extend(['best'] + list(objective_names))
        for model_label in requested_model_labels:
            if model_label == 'best':
                indiv = report.survivors[0]
            elif model_label not in objective_names:
                raise Exception('nested.analyze: problem finding model_key: %s in history_file_path: %s' %
                                (model_label, history_file_path))
            else:
                indiv = report.specialists[model_label]
            model_id = indiv.model_id
            x = indiv.x
            ids_to_labels[model_id].add(model_label)
            ids_to_labels[model_id].add(str(model_id))
            ids_to_param_arrays[model_id] = x

        # first search survivors and specialists already loaded into OptimizationReport instance
        if len(requested_model_ids) > 0:
            for model_id in requested_model_ids:
                for indiv in report.survivors:
                    if model_id == indiv.model_id:
                        break
                else:
                    for label, indiv in report.specialists.items():
                        if model_id == indiv.model_id:
                            break
                    else:
                        continue
                ids_to_labels[model_id].add(str(model_id))
                requested_model_ids.remove(model_id)
                ids_to_param_arrays[model_id] = indiv.x

        # if any requested_model_ids remain, the full optimization history must be loaded (slower):
        if len(requested_model_ids) > 0:
            history = OptimizationHistory(file_path=history_file_path)
            pop_size = history.count // len(history.history)
            for model_id in requested_model_ids:
                gen = model_id // pop_size
                for indiv in history.history[gen]:
                    if model_id == indiv.model_id:
                        break
                else:
                    for indiv in history.failed[gen]:
                        if model_id == indiv.model_id:
                            break
                    else:
                        raise Exception('nested.analyze: problem finding model_id: %i in history_file_path: %s' %
                                        (model_id, history_file_path))
                ids_to_labels[model_id].add(str(model_id))
                ids_to_param_arrays[model_id] = indiv.x

        for export_key, model_id in enumerate(ids_to_labels):
            export_keys.append(str(export_key))
            param_arrays.append(ids_to_param_arrays[model_id])
            # add all additional labels associated with each model_id
            if model_id == report.survivors[0].model_id:
                ids_to_labels[model_id].add('best')
            for label, indiv in report.specialists.items():
                if model_id == indiv.model_id:
                    ids_to_labels[model_id].add(label)
            model_labels.append(list(ids_to_labels[model_id]))

    elif param_file_path is not None:
        if not os.path.isfile(param_file_path):
            raise Exception('nested.analyze: invalid param_file_path: %s' % param_file_path)
        legend['source'] = param_file_path
        model_param_dict = read_from_yaml(param_file_path)

        if len(requested_model_labels) < 1 or 'all' in requested_model_labels:
            requested_model_labels = list(model_param_dict.keys())

        for export_key, key in enumerate(requested_model_labels):
            if str(key) in model_param_dict:
                this_param_dict = model_param_dict[str(key)]
            elif str(key).isnumeric() and int(key) in model_param_dict:
                this_param_dict = model_param_dict[int(key)]
            else:
                raise RuntimeError('nested.analyze: problem finding model_key: %s in in param_file_path: %s' %
                                   (key, param_file_path))
            this_param_names = list(this_param_dict.keys())
            uncommon_keys = np.setxor1d(this_param_names, param_names)
            if len(uncommon_keys) > 0:
                raise KeyError('parameter_names for model_key: %s loaded from param_file_path: %s does not match the '
                               'parameter_names specified in the config_file: %s' %
                               (key, param_file_path, str(param_names)))
            this_param_array = param_dict_to_array(this_param_dict, param_names)
            param_arrays.append(this_param_array)
            export_keys.append(str(export_key))
            model_labels.append([str(key)])

    legend['model_labels'] = model_labels
    legend['export_keys'] = model_keys

    return param_arrays, model_labels, export_keys, legend


def nested_optimize_init_controller_context(context, config_file_path=None, history_file_path=None, param_file_path=None,
                                     x0_key=None, param_gen=None, label=None, output_dir=None, disp=False, **kwargs):
    """
    :param context: :class:'Context'
    :param config_file_path: str (path)
    :param history_file_path: str (path)
    :param param_file_path: str (path)
    :param x0_key: str
    :param param_gen: str
    :param label: str
    :param output_dir: str (dir)
    :param disp: bool
    """
    if config_file_path is not None:
        context.config_file_path = config_file_path
    if 'config_file_path' not in context() or context.config_file_path is None or \
            not os.path.isfile(context.config_file_path):
        raise Exception('nested: config_file_path specifying required parameters is missing or invalid.')
    config_dict = read_from_yaml(context.config_file_path)

    nested_analyze_config_controller_context(context, config_dict, label, output_dir, disp, **kwargs)

    # ParamGenClass points to the parameter generator class, while param_gen points to its name as a string
    if 'param_gen' in config_dict and config_dict['param_gen'] is not None:
        context.param_gen = config_dict['param_gen']
    else:
        context.param_gen = param_gen
    if 'param_gen_source' not in context():
        if 'param_gen_source' in config_dict:
            context.param_gen_source = config_dict['param_gen_source']
        else:
            context.param_gen_source = None
    try:
        if context.param_gen_source is None and context.param_gen in globals() and \
                callable(globals()[context.param_gen]):
            context.ParamGenClass = globals()[context.param_gen]
        else:
            m = importlib.import_module(context.param_gen_source)
            param_gen_class = getattr(m, context.param_gen)
            if callable(param_gen_class):
                context.ParamGenClass = param_gen_class
            else:
                raise Exception
    except:
        raise Exception('nested.optimize: parameter generator: %s must be imported and callable' %
                        context.param_gen)
    if 'param_gen_kwargs' in config_dict and config_dict['param_gen_kwargs'] is not None:
        context.param_gen_kwargs = config_dict['param_gen_kwargs']
    else:
        context.param_gen_kwargs = {}
    for key in context.param_gen_kwargs:
        if key in context.kwargs:
            val = context.kwargs.pop(key)
            context.param_gen_kwargs[key] = val

    if context.output_dir is None:
        output_dir_str = ''
    else:
        output_dir_str = context.output_dir + '/'
    if context.optimization_title is not None:
        optimization_title_str = '_%s' % context.optimization_title
    else:
        optimization_title_str = ''
    if context.label is not None:
        label_str = '_%s' % context.label
    else:
        label_str = ''
    if history_file_path is not None:
        context.history_file_path = history_file_path
    timestamp = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    if 'history_file_path' not in context() or context.history_file_path is None:
        context.history_file_path = '%s%s_nested_optimization_history_%s%s%s_%i.hdf5' % \
                                    (output_dir_str, timestamp, context.param_gen, optimization_title_str, label_str,
                                     uuid.uuid1())

    # save config_file copy
    config_file_name = context.config_file_path.split('/')[-1]
    config_file_copy_path = '{!s}{!s}{!s}_{!s}'.format(output_dir_str, timestamp, label_str, config_file_name)
    shutil.copy2(context.config_file_path, config_file_copy_path)

    if param_file_path is not None:
        context.param_file_path = param_file_path
    if 'param_file_path' in context() and context.param_file_path is not None:
        load_x0_from_param_file = True
    else:
        load_x0_from_param_file = False
    if x0_key is not None:
        context.x0_key = x0_key
    if 'x0_key' in context() and context.x0_key is not None:
        if not load_x0_from_param_file:
            raise Exception('nested.optimize: cannot load initial parameters with x0_key: %s without specifying a '
                            'param_file_path' % context.x0_key)
        param_arrays, model_labels, _, _ = \
            load_model_params(context.param_names, param_file_path=context.param_file_path, model_keys=[context.x0_key])
        context.x0 = param_array_to_dict(param_arrays[0], context.param_names)
        if disp:
            print('nested.optimize: loaded initial model parameters with the following labels: %s' % model_labels[0])
            sys.stdout.flush()
    elif load_x0_from_param_file:
        raise Exception('nested.optimize: cannot load initial parameters from param_file_path without specifying an '
                        'x0_key' % context.param_file_path)

    if 'x0' in context() and context.x0 is not None:
        for param_name in context.default_params:
            context.x0[param_name] = context.default_params[param_name]
        context.x0_dict = context.x0
        context.x0_array = param_dict_to_array(context.x0_dict, context.param_names)


def nested_parallel_config_controller_context(context, config_dict=None, label=None, output_dir=None, disp=False,
                                              export_file_path=None, **kwargs):
    """

    :param context: :class:'Context'
    :param config_dict: dict
    :param label: str
    :param output_dir: str (dir)
    :param disp: bool
    :param export_file_path: str (path)
    """
    context.disp = disp
    if config_dict is None:
        config_dict = {}

    if 'kwargs' in config_dict and config_dict['kwargs'] is not None:
        context.kwargs = config_dict['kwargs']  # Extra arguments to be passed to imported sources
    else:
        context.kwargs = {}
    context.kwargs.update(kwargs)
    context.update(context.kwargs)

    if label is not None:
        context.label = label
    elif 'label' not in context():
        context.label = None
    if output_dir is not None:
        context.output_dir = output_dir
    if 'output_dir' not in context():
        context.output_dir = None
    if 'optimization_title' in config_dict:
        context.optimization_title = config_dict['optimization_title']
    if 'optimization_title' not in context():
        context.optimization_title = None

    if context.output_dir is None:
        output_dir_str = ''
    else:
        output_dir_str = context.output_dir + '/'
    if context.optimization_title is not None:
        optimization_title_str = '_%s' % context.optimization_title
    else:
        optimization_title_str = ''
    if context.label is not None:
        label_str = '_%s' % context.label
    else:
        label_str = ''

    if export_file_path is not None:
        context.export_file_path = export_file_path
    if 'export_file_path' not in context() or context.export_file_path is None:
        timestamp = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
        context.export_file_path = '%s%s_nested_exported_output%s%s_%i.hdf5' % \
                                   (output_dir_str, timestamp, optimization_title_str, label_str, uuid.uuid1())

    if 'interface' in context():
        if hasattr(context.interface, 'controller_comm'):
            context.controller_comm = context.interface.controller_comm
        if hasattr(context.interface, 'global_comm'):
            context.global_comm = context.interface.global_comm
        if hasattr(context.interface, 'num_workers'):
            context.num_workers = context.interface.num_workers
    if 'controller_comm' not in context():
        try:
            from mpi4py import MPI
            context.controller_comm = MPI.COMM_SELF
        except:
            pass


def nested_analyze_config_controller_context(context, config_dict, label=None, output_dir=None, disp=False,
                                             export_file_path=None, **kwargs):
    """

    :param context: :class:'Context'
    :param config_dict: dict
    :param label: str
    :param output_dir: str (dir)
    :param disp: bool
    :param export_file_path: str (path)
    """
    nested_parallel_config_controller_context(context, config_dict, label, output_dir, disp, export_file_path, **kwargs)

    if 'param_names' not in config_dict or config_dict['param_names'] is None:
        context.param_names = None
    else:
        context.param_names = config_dict['param_names']
    if 'default_params' not in config_dict or config_dict['default_params'] is None:
        context.default_params = {}
    else:
        context.default_params = config_dict['default_params']
    if 'bounds' not in config_dict or config_dict['bounds'] is None:
        context.bounds = None
    else:
        for param in context.default_params:
            config_dict['bounds'][param] = (context.default_params[param], context.default_params[param])
        if context.param_names is None:
            context.param_names = sorted(list(config_dict['bounds'].keys()))
        context.bounds = [config_dict['bounds'][key] for key in context.param_names]
    if 'rel_bounds' not in config_dict or config_dict['rel_bounds'] is None:
        context.rel_bounds = None
    else:
        context.rel_bounds = config_dict['rel_bounds']

    if 'target_val' in config_dict:
        context.target_val = config_dict['target_val']
    else:
        context.target_val = None
    if 'target_range' in config_dict:
        context.target_range = config_dict['target_range']
    else:
        context.target_range = None

    context.sources = []
    if 'config_collective' not in config_dict or config_dict['config_collective'] is None:
        context.config_collective_list = []
    else:
        context.config_collective_list = config_dict['config_collective']
    for item in context.config_collective_list:
        if 'source' not in item:
            raise Exception('nested: config_file at path: %s includes a config_collective function without a '
                            'source' % context.config_file_path)
        elif item['source'] not in context.sources:
            context.sources.append(item['source'])
    if 'update_context' not in config_dict or config_dict['update_context'] is None:
        context.update_context_list = []
    else:
        context.update_context_list = config_dict['update_context']
    for item in context.update_context_list:
        if 'source' not in item:
            raise Exception('nested: config_file at path: %s includes an update_context function without a '
                            'source' % context.config_file_path)
        elif item['source'] not in context.sources:
            context.sources.append(item['source'])
    if 'stages' not in config_dict or config_dict['stages'] is None:
        raise Exception('nested: config_file at path: %s is missing the required field: stages' %
                        context.config_file_path)
    else:
        context.stages = config_dict['stages']
    for item in context.stages:
        if 'source' not in item:
            raise Exception('nested: config_file: %s includes a stage of computation without a source' %
                            context.config_file_path)
        elif item['source'] not in context.sources:
            context.sources.append(item['source'])

    context.reset_worker_funcs = []
    context.shutdown_worker_funcs = []
    for source in context.sources:
        m = importlib.import_module(source)
        m_context_name = find_context_name(source)
        setattr(m, m_context_name, context)
        if hasattr(m, 'reset_worker'):
            reset_func = getattr(m, 'reset_worker')
            if not callable(reset_func):
                raise Exception('nested: reset_worker for source: %s is not a callable function.' % source)
            context.reset_worker_funcs.append(reset_func)
        if hasattr(m, 'shutdown_worker'):
            shutdown_func = getattr(m, 'shutdown_worker')
            if not callable(shutdown_func):
                raise Exception('nested: shutdown_worker for source: %s is not a callable function.' % source)
            context.shutdown_worker_funcs.append(shutdown_func)
        if hasattr(m, 'config_controller'):
            config_func = getattr(m, 'config_controller')
            if not callable(config_func):
                raise Exception('nested: config_controller for source: %s is not a callable function' % source)
            config_func()

    if 'bounds' not in context() or context.bounds is None or len(context.bounds) == 0:
        raise Exception('nested: bounds must either be specified within the provided config_file at path: %s, '
                        'or configured by a config_controller function.' % context.config_file_path)

    if 'param_names' not in context() or context.param_names is None or len(context.param_names) == 0:
        raise Exception('nested: param_names must either be inferred from bounds specified in the provided '
                        'config_file at path: %s, or configured by a config_controller function.' %
                        context.config_file_path)

    if 'x0' in config_dict and config_dict['x0'] is not None:
        context.x0 = config_dict['x0']
        for param_name in context.default_params:
            context.x0[param_name] = context.default_params[param_name]
    else:
        context.x0 = None
    context.x0_dict = context.x0
    if context.x0_dict is None:
        context.x0_array = None
    else:
        context.x0_array = param_dict_to_array(context.x0_dict, context.param_names)

    if 'feature_names' not in context():
        if 'feature_names' in config_dict:
            context.feature_names = config_dict['feature_names']
        else:
            context.feature_names = None
    if 'objective_names' not in context():
        if 'objective_names' in config_dict:
            context.objective_names = config_dict['objective_names']
        else:
            context.objective_names = None

    context.update_context_funcs = []
    for item in context.update_context_list:
        source = item['source']
        func_name = item['function']
        module = sys.modules[source]
        func = getattr(module, func_name)
        if not callable(func):
            raise Exception('nested: update_context: %s for source: %s is not a callable function.'
                            % (func_name, source))
        context.update_context_funcs.append(func)

    context.config_collective_funcs = []
    for item in context.config_collective_list:
        source = item['source']
        func_name = item['function']
        module = sys.modules[source]
        func = getattr(module, func_name)
        if not callable(func):
            raise Exception('nested: config_collective: %s for source: %s is not a callable function.'
                            % (func_name, source))
        context.config_collective_funcs.append(func)

    for stage in context.stages:
        source = stage['source']
        module = sys.modules[source]
        if 'get_args_static' in stage and stage['get_args_static'] is not None:
            func_name = stage['get_args_static']
            func = getattr(module, func_name)
            if not callable(func):
                raise Exception('nested: get_args_static: %s for source: %s is not a callable function.'
                                % (func_name, source))
            stage['get_args_static_func'] = func
        elif 'get_args_dynamic' in stage and stage['get_args_dynamic'] is not None:
            func_name = stage['get_args_dynamic']
            func = getattr(module, func_name)
            if not callable(func):
                raise Exception('nested: get_args_dynamic: %s for source: %s is not a callable function.'
                                % (func_name, source))
            stage['get_args_dynamic_func'] = func
        if 'compute_features' in stage and stage['compute_features'] is not None:
            func_name = stage['compute_features']
            func = getattr(module, func_name)
            if not callable(func):
                raise Exception('nested: compute_features: %s for source: %s is not a callable function.'
                                % (func_name, source))
            stage['compute_features_func'] = func
        elif 'compute_features_shared' in stage and stage['compute_features_shared'] is not None:
            func_name = stage['compute_features_shared']
            func = getattr(module, func_name)
            if not callable(func):
                raise Exception('nested: compute_features_shared: %s for source: %s is not a callable '
                                'function.' % (func_name, source))
            stage['compute_features_shared_func'] = func
        if 'filter_features' in stage and stage['filter_features'] is not None:
            func_name = stage['filter_features']
            func = getattr(module, func_name)
            if not callable(func):
                raise Exception('nested: filter_features: %s for source: %s is not a callable function.'
                                % (func_name, source))
            stage['filter_features_func'] = func
        if 'collective' in stage and stage['collective'] is not None:
            func_name = stage['collective']
            func = getattr(module, func_name)
            if not callable(func):
                raise Exception('nested: collective: %s for source: %s is not a callable function.'
                                % (func_name, source))
            stage['collective_func'] = func
        if 'get_objectives' in stage and stage['get_objectives'] is not None:
            func_name = stage['get_objectives']
            func = getattr(module, func_name)
            if not callable(func):
                raise Exception('nested: get_objectives: %s for source: %s is not a callable function.'
                                % (func_name, source))
            stage['get_objectives_func'] = func


def nested_analyze_init_controller_context(context, config_file_path=None, label=None, output_dir=None, disp=False,
                                           export_file_path=None, **kwargs):
    """
    :param context: :class:'Context'
    :param config_file_path: str (path)
    :param label: str
    :param output_dir: str (path)
    :param disp: bool
    :param export_file_path: str (path)
    """
    if config_file_path is not None:
        context.config_file_path = config_file_path
    if 'config_file_path' not in context() or context.config_file_path is None or \
            not os.path.isfile(context.config_file_path):
        raise Exception('nested: config_file_path specifying required parameters is missing or invalid.')
    config_dict = read_from_yaml(context.config_file_path)

    nested_analyze_config_controller_context(context, config_dict, label, output_dir, disp, export_file_path, **kwargs)


def nested_analyze_init_worker_contexts(sources, update_context_funcs, param_names, default_params, feature_names,
                                        objective_names, target_val, target_range, label, output_dir, disp, **kwargs):
    """

    :param sources: set of str (source names)
    :param update_context_funcs: list of callable
    :param param_names: list of str
    :param default_params: dict
    :param feature_names: list of str
    :param objective_names: list of str
    :param target_val: dict
    :param target_range: dict
    :param label: str
    :param output_dir: str (dir path)
    :param disp: bool
    """
    context = find_context()
    context.update(locals())
    context.update(kwargs)

    nested_parallel_config_worker_contexts(context, sources)


def nested_parallel_init_worker_contexts(sources, label=None, output_dir=None, disp=None, **kwargs):
    """

    :param sources: set of str (source names)
    :param output_dir: str (dir path)
    :param disp: bool
    :param label: str
    """
    context = find_context()
    context.update(locals())
    context.update(kwargs)

    nested_parallel_config_worker_contexts(context, sources)


def nested_parallel_config_worker_contexts(context, sources):
    """

    :param context: :class:'Context'
    :param sources: list
    """
    if context.label is not None:
        label_str = '_%s' % context.label
    else:
        label_str = ''

    if context.output_dir is not None:
        output_dir_str = context.output_dir + '/'
    else:
        output_dir_str = ''

    context.temp_output_path = '%s%s_nested_temp_output%s_pid%i_uuid%i.hdf5' % \
                               (output_dir_str, datetime.datetime.today().strftime('%Y%m%d_%H%M%S'), label_str,
                                os.getpid(), uuid.uuid1())

    if 'interface' in context():
        if hasattr(context.interface, 'comm'):
            context.comm = context.interface.comm
        if hasattr(context.interface, 'worker_comm'):
            context.worker_comm = context.interface.worker_comm
        if hasattr(context.interface, 'global_comm'):
            context.global_comm = context.interface.global_comm
        if hasattr(context.interface, 'num_workers'):
            context.num_workers = context.interface.num_workers
    if 'comm' not in context():
        try:
            from mpi4py import MPI
            context.comm = MPI.COMM_WORLD
        except Exception:
            pass

    for source_file in sources:
        source_dir = os.path.dirname(os.path.abspath(source_file))
        sys.path.insert(0, source_dir)
        source = os.path.basename(source_file).split('.py')[0]
        m = importlib.import_module(source)
        m_context_name = find_context_name(source)
        setattr(m, m_context_name, context)
        if hasattr(m, 'config_worker'):
            config_func = getattr(m, 'config_worker')
            if not callable(config_func):
                raise Exception('nested: init_worker_contexts: source: %s; problem executing config_worker' %
                                source)
            config_func()
    sys.stdout.flush()


def nested_analyze_init_contexts_interactive(context, config_file_path, label=None, output_dir=None, disp=None,
                                     export_file_path=None, history_file_path=None, param_file_path=None,
                                     model_key=None, model_id=None, **kwargs):
    """
    nested.analyze and nested.optimize are meant to be executed as modules, and refer to a config_file to import
    required submodules and create a workflow for model evaluation and optimization. During development of submodules,
    it is useful to be able to execute a submodule as a standalone script (as '__main__').
    nested_analyze_init_contexts_interactive allows a single process to properly parse the config_file and initialize
    the context on controller and worker processes for testing purposes.
    :param context: :class:'Context'
    :param config_file_path: str (.yaml file path)
    :param label: str
    :param output_dir: str (dir path)
    :param disp: bool
    :param export_file_path: str (.hdf5 file path)
    :param history_file_path: str (path)
    :param param_file_path: str (path)
    :param model_key: str
    :param model_id: int
    """
    nested_analyze_init_controller_context(context, config_file_path, label, output_dir, disp, export_file_path,
                                           **kwargs)

    if model_key is not None:
        model_key = [model_key]
    if model_id is not None:
        model_id = [model_id]
    param_arrays, model_labels, export_keys, legend = \
        load_model_params(context.param_names, param_file_path=param_file_path,
                          history_file_path=history_file_path, model_keys=model_key, model_ids=model_id)
    if len(param_arrays) < 1:
        if context.x0_array is None:
            raise Exception('nested: parameters must be specified either through a config_file, a param_file, or a '
                            'history_file')
    else:
        context.x0_array = param_arrays[0]
        context.x0_dict = param_array_to_dict(context.x0_array, context.param_names)
        context.x0 = context.x0_dict
        if disp:
            print('nested: evaluating model with the following labels: %s' % model_labels[0])
            sys.stdout.flush()

    context.interface.apply(nested_analyze_init_worker_contexts, context.sources, context.update_context_funcs,
                            context.param_names, context.default_params, context.feature_names,
                            context.objective_names, context.target_val, context.target_range, context.label,
                            context.output_dir, context.disp, **context.kwargs)

    for config_collective_func in context.config_collective_funcs:
        context.interface.collective(config_collective_func)


def nested_parallel_init_contexts_interactive(context, config_file_path=None, label=None, output_dir=None, disp=False,
                                              export_file_path=None, **kwargs):
    """
    nested.parallel is used for parallel map operations. This method imports optional parameters from a config_file and
    initializes a Context object on each worker.
    :param config_file_path: str (.yaml file path)
    :param label: str
    :param output_dir: str (dir path)
    :param disp: bool
    :param export_file_path: str (.hdf5 file path)
    """
    if config_file_path is not None:
        context.config_file_path = config_file_path
    if 'config_file_path' in context() and context.config_file_path is not None:
        if not os.path.isfile(context.config_file_path):
            raise Exception('nested.parallel: invalid (optional) config_file_path: %s' % context.config_file_path)
        else:
            config_dict = read_from_yaml(context.config_file_path)
    else:
        config_dict = {}
    context.update(config_dict)

    nested_parallel_config_controller_context(context, config_dict, label, output_dir, disp, export_file_path, **kwargs)

    # push all items at the top level of the config_dict to the worker context
    context.kwargs.update(config_dict)

    # overwrite items in the config_dict with kwargs passed through the command line
    context.kwargs.update(kwargs)

    m = sys.modules['__main__']
    local_source = m.__file__
    sources = [local_source]
    if hasattr(m, 'config_controller'):
        config_func = getattr(m, 'config_controller')
        if not callable(config_func):
            raise Exception('nested.parallel: source: %s; config_parallel_interface: problem executing '
                            'config_controller' % local_source)
        config_func()

    context.interface.apply(nested_parallel_init_worker_contexts, sources, context.label, context.output_dir,
                            context.disp, **context.kwargs)


def merge_exported_data(context, export_file_path=None, output_dir=None, legend=None, verbose=False):
    """

    :param export_file_path: str (path)
    :param output_dir: str (dir)
    :param legend: dict
    :param verbose: bool
    :return: str (path)
    """
    temp_output_path_list = [temp_output_path for temp_output_path in context.interface.get('context.temp_output_path')
                             if os.path.isfile(temp_output_path)]
    if len(temp_output_path_list) == 0:
        return None

    export_file_path = \
        merge_hdf5_temp_output_files(temp_output_path_list, export_file_path, output_dir=output_dir, verbose=verbose)
    for temp_output_path in temp_output_path_list:
        os.remove(temp_output_path)
    if legend is not None:
        with h5py.File(export_file_path, 'a') as f:
            set_h5py_attr(f.attrs, 'source', legend['source'])
            set_h5py_attr(f.attrs, 'model_labels', legend['model_labels'])
            set_h5py_attr(f.attrs, 'export_keys', legend['export_keys'])
    return export_file_path


def write_merge_path_list_to_yaml(context, export_file_path=None, output_dir=None, verbose=False):
    """

    :param context: :class:'Context'
    :param target: str (path)
    :param output_dir: str (dir)
    :param verbose: bool
    """
    if export_file_path is None:
        if output_dir is None or not os.path.isdir(output_dir):
            raise RuntimeError('write_merge_path_list_to_yaml: invalid output_dir: %s' % str(output_dir))
        export_file_path = '%s/%s_merged_exported_data_%i.hdf5' % \
                           (output_dir, datetime.datetime.today().strftime('%Y%H%M%S_%m%d'), os.getpid())

    temp_output_path_list = [temp_output_path for temp_output_path in
                             context.interface.get('context.temp_output_path')
                             if os.path.isfile(temp_output_path)]
    if len(temp_output_path_list) > 0:
        merge_file_path = '%s.yaml' % context.export_file_path.split('.hdf5')[0]
        data = {'export_file_path': export_file_path, 'temp_output_path_list': temp_output_path_list}
        write_to_yaml(merge_file_path, data)
        if verbose:
            print('write_merge_path_list_to_yaml: merge_file_path_list exported to %s' % merge_file_path)
            sys.stdout.flush()
    else:
        if verbose:
            print('write_merge_path_list_to_yaml: merge_path_list not exported; empty temp_output_path_list')
            sys.stdout.flush()


def merge_hdf5_temp_output_files(file_path_list, export_file_path=None, output_dir=None, verbose=False, debug=False):
    """
    When evaluating models with nested.analyze, each worker can export data to its own unique .hdf5 file
    (temp_output_path). Then the master process collects and merges these files into a single file (export_file_path).
    To avoid redundancy, this method only copies the top-level group 'shared_context' once. Then, the content of any
    other top-level groups are copied recursively. If a group attribute 'enumerated' exists and is True, this method
    expects data to be nested in groups enumerated with str(int) as keys. These data structures will be re-enumerated
    during the merge. Otherwise, groups containing nested data are expected to be labeled with unique keys, and nested
    structures are only copied once.
    :param file_path_list: list of str (paths)
    :param export_file_path: str (path)
    :param output_dir: str (dir)
    :param verbose: bool
    :param debug: bool
    :return str (path)
    """
    if export_file_path is None:
        if output_dir is None or not os.path.isdir(output_dir):
            raise RuntimeError('merge_hdf5_temp_output_files: invalid output_dir: %s' % str(output_dir))
        export_file_path = '%s/%s_merged_exported_data_%i.hdf5' % \
                           (output_dir, datetime.datetime.today().strftime('%Y%H%M%S_%m%d'), os.getpid())
    if not len(file_path_list) > 0:
        if verbose:
            print('merge_hdf5_temp_output_files: no data exported; empty file_path_list')
            sys.stdout.flush()
        return None
    
    start_time = time.time()
    with h5py.File(export_file_path, 'a') as new_f:
        for old_file_path in file_path_list:
            current_time = time.time()
            with h5py.File(old_file_path, 'r') as old_f:
                for group in old_f:
                    nested_merge_hdf5_groups(old_f[group], group, new_f, debug=debug)
            if verbose:
                print('merge_hdf5_temp_output_files: merging %s into %s took %.1f s' % 
                      (old_file_path, export_file_path, time.time() - current_time))
                sys.stdout.flush()

    if verbose:
        print('merge_hdf5_temp_output_files: merging temp output files into %s took %.1f s' %
              (export_file_path, time.time() - start_time))
        sys.stdout.flush()
    return export_file_path


def nested_merge_hdf5_groups(source, target_key, target, debug=False):
    """

    :param source: :class: in ['h5py.File', 'h5py.Group', 'h5py.Dataset']
    :param target_key: str
    :param target: :class: in ['h5py.File', 'h5py.Group']
    :param debug: bool
    """
    if target_key not in target:
        try:
            target.copy(source, target_key)
        except (IOError, AttributeError):
            pass
        return
    elif isinstance(source, h5py.Dataset) or target_key == 'shared_context':
        if debug:
            print('nested_merge_hdf5_groups: source: %s; target_key: %s not copied; already exists in target: %s' %
                  (source, target_key, target))
            sys.stdout.flush()
        return
    else:
        target = target[target_key]
        if 'enumerated' in source.attrs and source.attrs['enumerated']:
            count = len(target)
            for key in source:
                nested_merge_hdf5_groups(source[key], str(count), target, debug=debug)
                count += 1
            if debug:
                print('nested_merge_hdf5_groups: merged enumerated groups to target: %s' % target)
                sys.stdout.flush()
        else:
            for key in source:
                nested_merge_hdf5_groups(source[key], key, target, debug=debug)


def h5_nested_copy(source, target):
    """

    :param source: :class: in ['h5py.File', 'h5py.Group', 'h5py.Dataset']
    :param target: :class: in ['h5py.File', 'h5py.Group']
    """
    if isinstance(source, h5py.Dataset):
        try:
            target.copy(source, target)
        except (IOError, AttributeError):
            pass
        return
    else:
        for key, val in source.items():
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


def generate_sobol_seq(config_file_path, n, param_file_path):
    """
    uniform sampling with some randomness/jitter. generates n * (2d + 2) sets of parameter values, d being
        the number of parameters
    """
    from SALib.sample import saltelli
    from nested.utils import read_from_yaml
    from nested.lsa import get_param_bounds
    yaml_dict = read_from_yaml(config_file_path)
    bounds = get_param_bounds(config_file_path)
    problem = {
        'num_vars': len(yaml_dict['param_names']),
        'names': yaml_dict['param_names'],
        'bounds': bounds,
    }
    param_array = saltelli.sample(problem, n)
    save_pregen(param_array, param_file_path)
    return param_array


def sobol_analysis(config_file_path, history, jupyter=False, feat=True):
    """
    confidence intervals are inferred by bootstrapping, so they may change from
    run to run even if the param values are the same
    :param config_file_path: str to .yaml
    :param history: OptimizationHistory object
    :param jupyter: bool
    :param feat: if False, analysis on objectives. only relevant if jupyter is True, else both
        analyses will be done
    """
    print('nested: performing Sobol parameter sensitivity analysis...')
    sys.stdout.flush()
    from nested.utils import read_from_yaml
    from nested.lsa import get_param_bounds

    yaml_dict = read_from_yaml(config_file_path)
    bounds = get_param_bounds(config_file_path)
    param_names = yaml_dict['param_names']
    feature_names, objective_names = yaml_dict['feature_names'], yaml_dict['objective_names']

    problem = {
        'num_vars': len(param_names),
        'names': param_names,
        'bounds': bounds,
    }

    if not jupyter:
        sobol_analysis_helper('f', history, param_names, feature_names, problem)
        sobol_analysis_helper('o', history, param_names, objective_names, problem)
    elif feat:   # unable to do analyses in succession in jupyter
        return sobol_analysis_helper('f', history, param_names, feature_names, problem)
    else:
        return sobol_analysis_helper('o', history, param_names, objective_names, problem)


def sobol_analysis_helper(y_str, history, param_names, output_names, problem):
    from SALib.analyze import sobol
    from nested.lsa import pop_to_matrix, SobolPlot
    txt_path = 'data/{}_sobol_analysis_{}{}{}{}{}{}.txt'.format(y_str, *time.localtime())
    total_effects = np.zeros((len(param_names), len(output_names)))
    total_effects_conf = np.zeros((len(param_names), len(output_names)))
    first_order = np.zeros((len(param_names), len(output_names)))
    first_order_conf = np.zeros((len(param_names), len(output_names)))
    second_order = {}
    second_order_conf = {}

    X, y = pop_to_matrix(history, 'p', y_str, ['p'], ['o'])
    num_failed = sum([len(gen) for gen in history.failed])
    if history.total_models == 0:
        warnings.warn("Sobol analysis: All models failed and were not evaluated. Skipping "
                      "analysis of %s." % ('features' if y_str == 'f' else 'objectives'), Warning)
        return
    elif num_failed != 0:
        if y_str == 'f':
            warnings.warn("Sobol analysis: Some models failed and were not evaluated. Skipping "
                          "analysis of features.", Warning)
            return
        else:
            warnings.warn("Sobol analysis: Some models failed and were not evaluated. Setting "
                          "the objectives of these models to the max objectives.", Warning)
            X, y = default_failed_to_max(X, y, history)

    if y_str == 'f':
        print("\nFeatures:")
    else:
        print("\nObjectives:")
    for o in range(y.shape[1]):
        print("\n---------------Dependent variable {}---------------\n".format(output_names[o]))
        Si = sobol.analyze(problem, y[:, o], print_to_console=True)
        total_effects[:, o] = Si['ST']
        total_effects_conf[:, o] = Si['ST_conf']
        first_order[:, o] = Si['S1']
        first_order_conf[:, o] = Si['S1_conf']
        second_order[output_names[o]] = Si['S2']
        second_order_conf[output_names[o]] = Si['S2_conf']
        write_sobol_dict_to_txt(txt_path, Si, output_names[o], param_names)
    title = "Total effects - features" if y_str == 'f' else "Total effects - objectives"
    return SobolPlot(total_effects, total_effects_conf, first_order, first_order_conf, second_order, second_order_conf,
              param_names, output_names, err_bars=True, title=title)


def write_sobol_dict_to_txt(path, Si, y_name, input_names):
    """
    the dict returned from sobol.analyze is organized in a particular way
    :param path: str
    :param Si: dict
    :param y_name: str
    :param input_names: list of str
    :return:
    """
    with open(path, 'a') as f:
        f.write("\n---------------Dependent variable %s---------------\n" % y_name)
        f.write("Parameter S1 S1_conf ST ST_conf\n")
        for i in range(len(input_names)):
            f.write("%s %.6f %.6f %.6f %.6f\n"
                    % (input_names[i], Si['S1'][i], Si['S1_conf'][i], Si['ST'][i], Si['ST_conf'][i]))
        f.write("\nParameter_1 Parameter_2 S2 S2_conf\n")
        for i in range(len(input_names) - 1):
            for j in range(i + 1, len(input_names)):
                f.write("%s %s %.6f %.6f\n"
                        % (input_names[i], input_names[j], Si['S2'][i][j], Si['S2_conf'][i][j]))
        f.write("\n")


def default_failed_to_max(X, y, history):
    # objectives only
    if len(history.max_objectives) == 0:
        raise RuntimeError("Max objectives not stored or loaded correctly.")
    max_objective = history.max_objectives[0]
    for possible in history.max_objectives:
        max_objective = np.maximum(max_objective, possible)

    for pop in history.failed:
        for indiv in pop:
            X = np.vstack((X, indiv.x))
            y = np.vstack((y, max_objective))
    return X, y


def save_pregen(matrix, save_path):
    with h5py.File(save_path, "w") as f:
        f.create_dataset('parameters', data=matrix)
    print("Saved pregenerated parameters to file: %s" % save_path)


def load_pregen(save_path):
    f = h5py.File(save_path, 'r')
    pregen_matrix = f['parameters'][:]
    f.close()
    return pregen_matrix






