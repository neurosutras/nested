"""
Library of functions and classes to support nested.optimize
"""
__author__ = 'Aaron D. Milstein and Grace Ng'
from nested.utils import *
import collections
from scipy._lib._util import check_random_state
from copy import deepcopy


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
            if (hasattr(param_names, '__getitem__') and hasattr(feature_names, '__getitem__') and
                    hasattr(objective_names, '__getitem__')):
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
        of survivors produced closest to, but before the qth iteration.
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
        evaluate(group)
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
        colors = list(cm.rainbow(np.linspace(0, 1, len(self.history))))
        for this_attr in ['fitness', 'energy', 'distance', 'survivor']:
            fig, axes = plt.subplots(1)
            for j, population in enumerate(self.history):
                axes.scatter([indiv.rank for indiv in population], [getattr(indiv, this_attr) for indiv in population],
                            c=colors[j], alpha=0.05)
                axes.scatter([indiv.rank for indiv in self.survivors[j]],
                            [getattr(indiv, this_attr) for indiv in self.survivors[j]], c=colors[j], alpha=0.5)
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
        fig, axes = plt.subplots(1)
        this_attr = 'objectives'
        for j, population in enumerate(self.history):
            axes.scatter([indiv.rank for indiv in population],
                        [np.sum(getattr(indiv, this_attr)) for indiv in population],
                        c=colors[j], alpha=0.05)
            axes.scatter([indiv.rank for indiv in self.survivors[j]],
                        [np.sum(getattr(indiv, this_attr)) for indiv in self.survivors[j]], c=colors[j], alpha=0.5)
        divider = make_axes_locatable(axes)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
        cbar.set_label('Generation')
        clean_axes(axes)
        axes.set_xlabel('Ranked individuals per iteration')
        axes.set_title('absolute energy')
        plt.show()
        plt.close()
        for i, param_name in enumerate(self.param_names):
            this_attr = 'x'
            fig, axes = plt.subplots(1)
            for j, population in enumerate(self.history):
                axes.scatter([indiv.rank for indiv in population],
                            [getattr(indiv, this_attr)[i] for indiv in population],
                            c=colors[j], alpha=0.05)
                axes.scatter([indiv.rank for indiv in self.survivors[j]],
                            [getattr(indiv, this_attr)[i] for indiv in self.survivors[j]], c=colors[j], alpha=0.5)
                axes.scatter([-1 for indiv in self.failed[j]],
                            [getattr(indiv, this_attr)[i] for indiv in self.failed[j]], c='k', alpha=0.5)
            axes.set_xlabel('Ranked individuals per iteration')
            axes.set_title(param_name)
            divider = make_axes_locatable(axes)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
            cbar.set_label('Generation')
            clean_axes(axes)
        plt.show()
        plt.close()
        for i, objective_name in enumerate(self.objective_names):
            this_attr = 'objectives'
            fig, axes = plt.subplots(1)
            for j, population in enumerate(self.history):
                axes.scatter([indiv.rank for indiv in population],
                            [getattr(indiv, this_attr)[i] for indiv in population],
                            c=colors[j], alpha=0.05)
                axes.scatter([indiv.rank for indiv in self.survivors[j]],
                            [getattr(indiv, this_attr)[i] for indiv in self.survivors[j]], c=colors[j], alpha=0.5)
            axes.set_title(this_attr+': '+objective_name)
            axes.set_xlabel('Ranked individuals per iteration')
            divider = make_axes_locatable(axes)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
            cbar.set_label('Generation')
            clean_axes(axes)
        plt.show()
        plt.close()
        for i, feature_name in enumerate(self.feature_names):
            this_attr = 'features'
            fig, axes = plt.subplots(1)
            for j, population in enumerate(self.history):
                axes.scatter([indiv.rank for indiv in population],
                            [getattr(indiv, this_attr)[i] for indiv in population],
                            c=colors[j], alpha=0.05)
                axes.scatter([indiv.rank for indiv in self.survivors[j]],
                            [getattr(indiv, this_attr)[i] for indiv in self.survivors[j]], c=colors[j], alpha=0.5)
            axes.set_title(this_attr+': '+feature_name)
            axes.set_xlabel('Ranked individuals per iteration')
            divider = make_axes_locatable(axes)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
            cbar.set_label('Generation')
            clean_axes(axes)
        plt.show()
        plt.close()

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
                offset = 10.**(this_order_mag - 2)

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
                offset = 10. ** (this_order_mag - 2)
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


class BoundedStep(object):
    """
    Step-taking method for use with PopulationAnnealing. Steps each parameter within specified bounds. Explores the
    range in log10 space when the range is >= 2 orders of magnitude. Uses the log-modulus transformation
    (John & Draper, 1980) as an approximation that tolerates ranges that span zero. If bounds are not provided for some
    parameters, the default is (0.1 * x0, 10. * x0).
    """
    def __init__(self, x0, bounds=None, stepsize=0.5, wrap=False, random=None, **kwargs):
        """

        :param x0: array
        :param bounds: list of tuple
        :param stepsize: float in [0., 1.]
        :param wrap: bool  # whether or not to wrap around bounds
        :param random: int or :class:'np.random.RandomState'
        """
        self.wrap = wrap
        self.stepsize = stepsize
        if x0 is None and bounds is None:
            raise ValueError('BoundedStep: Either starting parameters or bounds are missing.')
        if random is None:
            self.random = np.random
        else:
            self.random = random
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
                    raise ValueError('BoundedStep: Either starting parameters or bounds are missing.')
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
        self.order_mag = np.ones_like(self.x0)
        if np.any(self.xmin == self.xmax):
            raise ValueError('BoundedStep: xmin and xmax cannot have the same value.')
        for i in xrange(len(self.x0)):
            if self.xmin[i] == 0. and self.xmax[i] != 0.:
                self.order_mag[i] = abs(np.log10(abs(self.xmax[i])))
            elif self.xmax[i] == 0. and self.xmin[i] != 0.:
                self.order_mag[i] = abs(np.log10(abs(self.xmin[i])))
            else:
                self.order_mag[i] = abs(np.log10(abs(self.xmax[i] / self.xmin[i])))
        self.logmod = lambda x: np.sign(x) * np.log10(np.add(np.abs(x), 1.))
        self.logmod_inv = lambda x: np.sign(x) * ((10. ** np.abs(x)) - 1.)
        self.logmod_xmin = self.logmod(self.xmin)
        self.logmod_xmax = self.logmod(self.xmax)
        self.logmod_range = np.subtract(self.logmod_xmax, self.logmod_xmin)

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
            if self.order_mag[i] >= 2.:
                x[i] = self.log10_step(x[i], i, stepsize, wrap)
            else:
                x[i] = self.linear_step(x[i], i, stepsize, wrap)
        return x

    def linear_step(self, xi, i, stepsize=None, wrap=None):
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
        if wrap:
            delta = self.random.uniform(-step, step)
            new_xi = xi + delta
            if self.xmin[i] > new_xi:
                new_xi = self.xmax[i] - (self.xmin[i] - new_xi)
            elif self.xmax[i] < new_xi:
                new_xi = self.xmin[i] + (new_xi - self.xmax[i])
        else:
            xi_min = max(self.xmin[i], xi - step)
            xi_max = min(self.xmax[i], xi + step)
            new_xi = self.random.uniform(xi_min, xi_max)
        return new_xi

    def log10_step(self, xi, i, stepsize=None, wrap=None):
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
        step = stepsize * self.logmod_range[i] / 2.
        logmod_xi = self.logmod(xi)
        if wrap:
            delta = np.random.uniform(-step, step)
            new_logmod_xi = logmod_xi + delta
            if self.logmod_xmin[i] > new_logmod_xi:
                new_logmod_xi = self.logmod_xmax[i] - (self.logmod_xmin[i] - new_logmod_xi)
            elif self.logmod_xmax[i] < new_logmod_xi:
                new_logmod_xi = self.logmod_xmin[i] + (new_logmod_xi - self.logmod_xmax[i])
        else:
            logmod_xi_min = max(self.logmod_xmin[i], logmod_xi - step)
            logmod_xi_max = min(self.logmod_xmax[i], logmod_xi + step)
            new_logmod_xi = self.random.uniform(logmod_xi_min, logmod_xi_max)
        new_xi = self.logmod_inv(new_logmod_xi)
        return new_xi

    def check_bounds(self, x):
        """

        :param x: array
        :return: bool
        """
        #check absolute bounds first
        for i, xi in enumerate(x):
            if not (xi == self.xmin[i] and xi == self.xmax[i]):
                if (xi < self.xmin[i]):
                    return False
                if (xi >= self.xmax[i]):
                    return False
        return True


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


def assign_relative_energy(population):
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
    for m in xrange(num_objectives):
        objective_vals = [individual.objectives[m] for individual in population]
        objective_min = min(objective_vals)
        objective_max = max(objective_vals)
        if objective_min != objective_max:
            objective_vals = np.subtract(objective_vals, objective_min)
            objective_vals = np.divide(objective_vals, objective_max - objective_min)
            for energy, individual in zip(objective_vals, population):
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


def evaluate_population_annealing(population, disp=False):
    """
    Modifies in place the fitness, energy and rank attributes of each Individual in the population.
    :param population: list of :class:'Individual'
    :param disp: bool
    """
    if len(population) > 0:
        assign_fitness_by_dominance(population)
        # assign_relative_energy_by_fitness(population)
        assign_relative_energy(population)
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
            self.select = select_survivors_by_rank_and_fitness  # select_survivors_by_rank
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
            if objective_dict is None or features[i] is None:
                self.failed.append(self.population[i])
                num_failed += 1
            elif type(objective_dict) != dict:
                raise TypeError('PopulationAnnealing.update_population: objectives must be a list of dict')
            elif type(features[i]) != dict:
                raise TypeError('PopulationAnnealing.update_population: features must be a list of dict')
            else:
                this_objectives = np.array([objective_dict[key] for key in self.storage.objective_names])
                self.population[i].objectives = this_objectives
                this_features = np.array([features[i][key] for key in self.storage.feature_names])
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
        survivors = self.select(candidate_survivors, self.num_survivors, max_fitness=self.max_fitness)
        for individual in survivors:
            individual.survivor = True
        self.survivors = survivors
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


class Evolution(object):
    """
    This class is inspired by emoo (Bahl A, Stemmler MB, Herz AVM, Roth A. (2012). J Neurosci Methods). It provides a
    generator interface to produce a list of parameter arrays for parallel evaluation.
    """

    def __init__(self, param_names=None, feature_names=None, objective_names=None, pop_size=None, x0=None,
                 bounds=None, rel_bounds=None, wrap_bounds=False, take_step=None, initial_step_size=1., m0=20, c0=20,
                 p_m=0.5, delta_m=0, delta_c=0, mutate_survivors=False, evaluate=None, seed=None, max_iter=None,
                 survival_rate=0.1,  disp=False, hot_start=None, **kwargs):
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
        :param m0: int : initial strength of mutation
        :param c0: int : initial strength of crossover
        :param p_m: float : probability of mutation
        :param delta_m: int : decrease mutation strength every interval
        :param delta_c: int : decrease crossover strength every interval
        :param mutate_survivors: bool
        :param evaluate: callable
        :param seed: int or :class:'np.random.RandomState'
        :param max_iter: int
        :param survival_rate: float in [0., 1.]
        :param disp: bool
        :param hot_start: str (path)
        :param kwargs: dict of additional options, catches generator-specific options that do not apply
        """
        if x0 is None:
            self.x0 = None
        else:
            self.x0 = np.array(x0)
        self.num_params = len(param_names)
        if evaluate is None:
            self.evaluate = evaluate_population_annealing
        elif isinstance(evaluate, collections.Callable):
            self.evaluate = evaluate
        else:
            raise TypeError("Evolution: evaluate must be callable.")
        self.random = check_random_state(seed)
        self.xmin = np.array([bound[0] for bound in bounds])
        self.xmax = np.array([bound[1] for bound in bounds])
        if hot_start is not None:
            if not os.path.isfile(hot_start):
                raise IOError('Evolution: invalid file path. Cannot hot start from stored history: %s' % hot_start)
            else:
                self.storage = PopulationStorage(file_path=hot_start)
                self.num_gen = len(self.storage.history)
                self.population = self.storage.history[-1]
                self.survivors = self.storage.survivors[-1]
                self.failed = self.storage.failed[-1]
                self.objectives_stored = True
        else:
            self.storage = PopulationStorage(param_names=param_names, feature_names=feature_names,
                                             objective_names=objective_names, path_length=1)
            self.num_gen = 0
            self.population = []
            self.survivors = []
            self.failed = []
            self.objectives_stored = False
        self.pop_size = pop_size
        if take_step is None:
            self.take_step = RelativeBoundedStep(self.x0, param_names=param_names, bounds=bounds, rel_bounds=rel_bounds,
                                                 stepsize=initial_step_size, wrap=wrap_bounds, random=self.random)
            self.x0 = np.array(self.take_step.x0)
            self.xmin = np.array(self.take_step.xmin)
            self.xmax = np.array(self.take_step.xmax)
        else:
            if take_step in globals() and callable(globals()[take_step]):
                self.take_step = globals()[take_step](self.x0, param_names=param_names, bounds=bounds,
                                                      rel_bounds=rel_bounds, stepsize=initial_step_size,
                                                      wrap=wrap_bounds, random=self.random)
                self.x0 = np.array(self.take_step.x0)
                self.xmin = np.array(self.take_step.xmin)
                self.xmax = np.array(self.take_step.xmax)
            else:
                raise TypeError('Evolution: provided take_step: %s is not callable.' % take_step)
        if max_iter is None:
            self.max_gens = 30.
        else:
            self.max_gens = max_iter
        self.num_survivors = max(1, int(self.pop_size * survival_rate))
        self.disp = disp
        self.evaluated = False
        self.local_time = time.time()
        self.m0 = m0
        self.m = self.m0
        self.c0 = c0
        self.c = self.c0
        self.p_m = p_m
        self.delta_m = delta_m
        self.delta_c = delta_c
        self.mutate_survivors = mutate_survivors

    def get_random_params(self):
        """

        :return: array
        """
        return np.random.uniform(self.param_min, self.param_max)

    def init_population(self):
        """

        """
        self.population = []
        for i in xrange(self.pop_size):
            params = self.random_params()
            individual = Individual(params)
            self.population.append(individual)
        self.evaluated = False

    def return_to_bounds(self, p):
        """

        :param p: array
        :return: array
        """
        p = np.minimum(p, self.param_max)
        p = np.maximum(p, self.param_min)
        p = self.take_step.apply_rel_bounds(self, p, stepsize, rel_bounds=None, disp=False)
        return p

    def evolve(self, maxgen=200):
        """
        Generator yields a new population. Requires that features have been evaluated and fitness assigned to current
        population.
        :param maxgen: int
        :yield: list of :class:'Individual'
        """
        self.current_gen = 0
        while self.current_gen < maxgen:
            if self.current_gen == 0:
                self.m = self.m0
                self.c = self.c0
                self.init_population()
                if self.interval is None:
                    self.interval = maxgen
                if self.verbose:
                    print 'Starting evolutionary multiobjective optimization generator (Evolution)\n'
                    print 'Based on Bahl A, Stemmler MB, Herz AVM, Roth A. (2012). J Neurosci Methods.\n'
                    print 'Modified by Aaron D. Milstein, Grace Ng, Ivan Soltesz (2017).'
                yield self.population
            elif not self.evaluated:
                raise Exception('Evolution step: evolution; fitness of current population has not been evaluated.')
            else:
                if self.current_gen % self.interval == 0:
                    self.m += self.delta_m
                    self.c += self.delta_c
                    if self.verbose:
                        print 'Generation %i/%i: Decreasing strength of mutation and crossover' % \
                              (self.current_gen, maxgen)
                self.selection()
                self.crossover()
                self.mutation()
                yield self.population
            self.current_gen += 1

            # self.evaluate()
            # self.assign_fitness()
            # if (self.checkpopulation != None):
            #    self.checkpopulation(self.population)
        self.report()

    def selection(self):
        """
        In this step the mating pool is formed by selection. The population is shuffled, each individual is compared to
        its neighbor, and the individual with high fitness score is transferred into the mating pool. This procedure is
        repeated twice.
        """
        if not self.evaluated:
            raise Exception('Evolution step: selection; Fitness of current population has not been evaluated.')

        mating_pool = []

        for k in xrange(2):
            population_permutation = self.population[np.random.permutation(len(self.population))]

            for i in np.arange(0, len(self.population) - 1, 2):
                individual1, individual2 = population_permutation[i], population_permutation[i + 1]
                if individual1.fitness < individual2.fitness:
                    mating_pool.append(individual1)
                else:
                    mating_pool.append(individual2)
        self.population = list(mating_pool)

    def crossover(self):
        """

        """
        children = []
        # do not add more children then original population size
        while len(children) + len(self.population) < 2 * self.pop_size:
            i, j = np.random.choice(range(len(self.population)), 2)
            parent1 = self.population[i]
            parent2 = self.population[j]
            child1_params = np.empty(self.num_params)
            child2_params = np.empty(self.num_params)
            for i in xrange(self.num_params):
                u_i = np.random.random()
                if u_i <= 0.5:
                    beta_q_i = pow(2. * u_i, 1. / (self.c + 1))
                else:
                    beta_q_i = pow(1. / (2. * (1. - u_i)), 1. / (self.c + 1))
                child1_params[i] = 0.5 * ((1. + beta_q_i) * parent1.p[i] + (1. - beta_q_i) * parent2.p[i])
                child2_params[i] = 0.5 * ((1. - beta_q_i) * parent1.p[i] + (1 + beta_q_i) * parent2.p[i])
            child1 = Individual(self.return_to_bounds(child1_params))
            child2 = Individual(self.return_to_bounds(child2_params))
            children.append(child1)
            children.append(child2)
        self.population.extend(children)

    def mutation(self):
        """
        polynomial mutation (Deb, 2001)
        """
        for k in xrange(len(self.population)):
            individual = self.population[k]
            if self.mutate_parents or individual.fitness is None:
                individual.fitness = None
                for i in xrange(self.num_params):
                    # each gene only mutates with a certain probability
                    if np.random.random() < self.p_m:
                        r_i = np.random.random()
                        if r_i < 0.5:
                            delta_i = pow(2. * r_i, 1. / (self.m + 1)) - 1.
                        else:
                            delta_i = 1. - pow(2. * (1. - r_i), 1. / (self.m + 1))
                        individual.p[i] += delta_i
                individual.p = self.return_to_bounds(individual.p)

    def evaluate(self):
        # only evaluate up to pop_size, as that number of processes must be pre-allocated
        new_population = []

        # is the master alone?
        if (self.mpi == False):

            for individual in self.population:

                # only evaluate those that are really new!
                if individual[self.fitnesspos] == -1:

                    parameters = individual[:self.para]

                    objectives_error = self.evaluate_individual(parameters)

                    if (objectives_error != None):
                        new_population.append(np.r_[parameters, objectives_error, self.no_properties])
                else:
                    new_population.append(individual)
        else:
            # distribute the individuals among the slaves
            i = 0
            for individual in self.population:
                if individual[self.fitnesspos] == -1:
                    parameters = individual[:self.para]

                    dest = i % (self.comm.size - 1) + 1
                    self.comm.send(parameters, dest=dest)
                    i += 1
                else:
                    new_population.append(individual)

            # the master does also one
            # TODO

            # Receive the results from the slaves
            for i in range(i):
                result = self.comm.recv(source=MPI.ANY_SOURCE)

                if result != None:
                    new_population.append(np.r_[result[0], result[1], self.no_properties])

        self.population = np.array(new_population)

    def evaluate_individual(self, parameters):

        parameters_unnormed = self.unnormit(parameters)

        # make a dictionary with the unormed parameters and send them to the evaluation function
        dict_parameters_normed = dict({})
        for i in range(len(self.variables)):
            dict_parameters_normed[self.variables[i][0]] = parameters_unnormed[i]

        dict_results = self.get_objectives_error(dict_parameters_normed)

        list_results = []
        for objective_name in self.objectives_names:
            list_results.append(dict_results[objective_name])

        for info_name in self.infos_names:
            list_results.append(dict_results[info_name])

        return np.array(list_results)

    def evaluate_slave(self):

        # We wait for parameters
        # we do not see the whole population!

        while (True):
            parameters = self.comm.recv(source=0)  # wait....

            # Does the master want the slave to shutdown?
            if (parameters == None):
                # Slave finishing...
                break

            objectives_error = self.evaluate_individual(parameters)

            # objectives_error = self.get_objectives_error(self.unnormit(parameters))
            if (objectives_error == None):
                self.comm.send(None, dest=0)
            else:
                self.comm.send([parameters, objectives_error], dest=0)

    def assign_fitness(self):
        """
        are we in a multiobjective regime, then the selection of the best individual is not trival
        and must be based on dominance, thus we determine all non dominated fronts and only use the best
        to transfer into the new generation
        """
        if (self.obj > 1):
            self.assign_rank()

            new_population = np.array([])

            maxrank = self.population[:, self.rankpos].max()

            for rank in range(0, int(maxrank) + 1):

                new_front = self.population[np.where(self.population[:, self.rankpos] == rank)]

                new_sorted_front = self.crowding_distance_sort(new_front)

                if (len(new_population) == 0):
                    new_population = new_sorted_front
                else:
                    new_population = np.r_[new_population, new_sorted_front]

            self.population = new_population

        else:
            # simple sort the objective value
            ind = np.argsort(self.population[:, self.objpos])
            self.population = self.population[ind]

        # now set the fitness, indiviauls are sorted, thus fitnes is easy to set
        fitness = range(0, len(self.population[:, 0]))
        self.population[:, -1] = fitness

    def new_generation(self):
        # the worst are at the end, let them die, if there are too many
        if (len(self.population) > self.size):
            self.population = self.population[:self.size]

    def dominates(self, p, q):

        objectives_error1 = self.population[p][self.objpos:self.objpos + self.obj]
        objectives_error2 = self.population[q][self.objpos:self.objpos + self.obj]

        diff12 = objectives_error1 - objectives_error2

        # is individdum equal or better then individdum two?
        # and at least in one objective better
        # then it dominates individuum2
        # if not it does not dominate two (which does not mean that 2 may not dominate 1)
        return (((diff12 <= 0).all()) and ((diff12 < 0).any()))

    def assign_rank(self):

        F = dict()

        P = self.population

        S = dict()
        n = dict()
        F[0] = []

        # determine how many solutions are dominated or dominate
        for p in range(len(P)):

            S[p] = []  # this is the list of solutions dominated by p
            n[p] = 0  # how many solutions are dominating p

            for q in range(len(P)):

                if self.dominates(p, q):
                    S[p].append(q)  # add q to the list of solutions dominated by p
                elif self.dominates(q, p):
                    n[p] += 1  # q dominates p, thus increase number of solutions that dominate p

            if n[p] == 0:  # no other solution dominates p

                # this is the rank column
                P[p][self.rankpos] = 0

                F[0].append(p)  # add p to the list of the first front

        # find the other non dominated fronts
        i = 0
        while len(F[i]) > 0:
            Q = []  # this will be the next front

            # take the elements from the last front
            for p in F[i]:

                # and take the elements that are dominated by p
                for q in S[p]:
                    # decrease domination number of all elements that are dominated by p
                    n[q] -= 1
                    # if the new domination number is zero, than we have found the next front
                    if n[q] == 0:
                        P[q][self.rankpos] = i + 1
                        Q.append(q)

            i += 1
            F[i] = Q  # this is the next front

    def crowding_distance_sort(self, front):

        sorted_front = front.copy()

        l = len(sorted_front[:, 0])

        sorted_front[:, self.distpos] = np.zeros_like(sorted_front[:, 0])

        for m in range(self.obj):
            ind = np.argsort(sorted_front[:, self.objpos + m])
            sorted_front = sorted_front[ind]

            # definitely keep the borders
            sorted_front[0, self.distpos] += 1000000000000000.
            sorted_front[-1, self.distpos] += 1000000000000000.

            fm_min = sorted_front[0, self.objpos + m]
            fm_max = sorted_front[-1, self.objpos + m]

            if fm_min != fm_max:
                for i in range(1, l - 1):
                    sorted_front[i, self.distpos] += (sorted_front[i + 1, self.objpos + m] - sorted_front[
                        i - 1, self.objpos + m]) / (fm_max - fm_min)

        ind = np.argsort(sorted_front[:, self.distpos])
        sorted_front = sorted_front[ind]
        sorted_front = sorted_front[-1 - np.arange(len(sorted_front))]

        return sorted_front


def merge_exported_data(file_path_list, new_file_path=None, verbose=True):
    """
    Each nested.optimize worker can export data intermediates to its own unique .hdf5 file (temp_output_path). Then the
    master process collects and merges these files into a single file (export_file_path). To avoid redundancy, this
    method only copies the top-level group 'shared_context' once. Then, the content of the top-level group
    'exported_data' is copied recursively. If a group attribute 'enumerated' exists and is True, this method expects
    data to be nested in groups enumerated with str(int) as keys. These data structures will be re-enumerated during
    the merge. Otherwise, groups containing nested data are expected to be labeled with unique keys, and nested
    structures are only copied once.
    :param file_path_list: list of str (paths)
    :param new_file_path: str (path)
    :return str (path)
    """
    if new_file_path is None:
        new_file_path = 'merged_hdf5_'+datetime.datetime.today().strftime('%m%d%Y%H%M')+'_'+os.getpid()
    if not len(file_path_list) > 0:
        return None
    enumerated = None
    enum = 0
    with h5py.File(new_file_path, 'w') as new_f:
        for old_file_path in file_path_list:
            with h5py.File(old_file_path, 'r') as old_f:
                if not 'shared_context' in new_f and 'shared_context' in old_f:
                    new_f.copy(old_f['shared_context'], new_f)
                if 'exported_data' in old_f:
                    if not 'exported_data' in new_f:
                        new_f.create_group('exported_data')
                        target = new_f['exported_data']
                    if enumerated is None:
                        if 'enumerated' in old_f.attrs and old_f.attrs['enumerated']:
                            enumerated = True
                            target.attrs['enumerated'] = True
                        else:
                            enumerated = False
                            target.attrs['enumerated'] = False
                    if enumerated:
                        for source in old_f.itervalues():
                            target.copy(source, target, name=str(enum))
                            enum += 1
                    else:
                        h5_nested_copy(old_f['exported_data'], target)
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
