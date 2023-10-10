import numpy as np
from collections import defaultdict
from scipy.stats import linregress, iqr
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
import os
from os import path
import warnings
import time
import io
from nested.optimize_utils import OptimizationReport, OptimizationHistory, normalize_dynamic, normalize_linear
from nested.utils import read_from_yaml
from collections import Iterable


class ParameterSensitivity(object):
    def __init__(self, file_path=None, history=None, parameters=None, features=None, objectives=None, param_names=None,
                 feature_names=None, objective_names=None):
        """
        Provide optimization history as:
            1. An .hdf5 file path
            2. An OptimizationHistory object
            3. 2D arrays of parameters, features, and objectives (num_samples, num_features), as well as lists of
                str names for parameters, features, and objectives, or a config_file_path to a .yaml file that defines
                them.
        example usage:
            history = OptimizationHistory(file_path="path.hdf5")
            sa = ParameterSensitivity(history=history)
            sa.plot_compare(x='parameter_name', y='feature_name')
            result = sa.compute_parameter_importances(plot=True)
        in a jupyter notebook:
            history = OptimizationHistory(file_path="path.hdf5")
            sa = ParameterSensitivity(history=history)
            sa.plot_interactive()

        :param file_path: str to path of .hdf5 file containing optimization history
        :param history: :class:'OptimizationHistory'
        :param parameters: 2d array of float
        :param features: 2d array of float
        :param objectives: 2d array of float
        :param param_names: list of str
        :param feature_names: list of str
        :param objective_names: list of str
        """
        if history is None and file_path is None:
            if any(category is None for category in [parameters, features, objectives]):
                raise Exception('ParameterSensitivity: please provide data either as a file_path, an'
                                'OptimizationHistory object, or 2D arrays for parameters, features, and objectives')
            else:
                self.parameters = parameters
                self.features = features
                self.objectives = objectives
                if any(name_list is None for name_list in [param_names, feature_names, objective_names]):
                    raise Exception('ParameterSensitivity: please provide str names for parameters, features, and '
                                    'objectives')
                self.param_names = param_names
                self.feature_names = feature_names
                self.objective_names = objective_names
        else:
            if history is None:
                history = OptimizationHistory(file_path=file_path)
            self.history = history
            self.param_names = self.history.param_names
            self.feature_names = self.history.feature_names
            self.objective_names = self.history.objective_names
            param_list, feature_list, objective_list = [], [], []
            for generation in self.history.generations:
                for indiv in generation:
                    param_list.append(indiv.x)
                    feature_list.append(indiv.features)
                    objective_list.append(indiv.objectives)
            self.parameters = np.array(param_list)
            self.features = np.array(feature_list)
            self.objectives = np.array(objective_list)
        self.linear_norm_parameters, self.linear_norm_features, self.linear_norm_objectives = None, None, None
        self.dynamic_norm_parameters, self.dynamic_norm_features, self.dynamic_norm_objectives = None, None, None
        self.min_parameters = np.min(self.parameters, axis=0)
        self.max_parameters = np.max(self.parameters, axis=0)
        self.min_features = np.min(self.features, axis=0)
        self.max_features = np.max(self.features, axis=0)
        self.min_objectives = np.min(self.objectives, axis=0)
        self.max_objectives = np.max(self.objectives, axis=0)

    def normalize(self, categories=None, norm='dynamic', threshold=2):
        """
        Normalize data between zero and one. Either scale linearly, or log scale if range is greater than threshold
        orders of magnitude.
        Allowable categories, either as a single str or a list of str: ['parameters', 'features', 'objectives']
        Allowable norm types as str: ['linear', 'dynamic']
        :param categories: str or list of str
        :param norm: str
        :param threshold
        """
        if categories is None:
            # defaults assume parameters with wide bounds and squared error objectives, whereas features are
            # typically measurements that the user may want to leave non-normalized
            categories = ['parameters', 'objectives']
        elif isinstance(categories, str):
            if categories in ['parameters', 'features', 'objectives']:
                categories = [categories]
            else:
                raise Exception('SensitivityAnalysis.normalize: invalid category: %s' % categories)
        elif isinstance(categories, Iterable):
            if not all([category in ['parameters', 'features', 'objectives'] for category in categories]):
                raise Exception('SensitivityAnalysis.normalize: invalid categories: %s' % categories)
        if norm not in ['dynamic', 'linear']:
            raise Exception('SensitivityAnalysis.normalize: invalid normalization type: %s' % norm)
        for category_name in categories:
            category = getattr(self, category_name)
            print(category_name, category)
            min_category = getattr(self, 'min_%s' % category_name)
            max_category = getattr(self, 'max_%s' % category_name)
            norm_category = np.empty_like(category)
            for col in range(category.shape[1]):
                vals = category[:,col]
                if norm == 'dynamic':
                    vals = normalize_dynamic(vals, min_category[col], max_category[col], threshold=threshold)
                elif norm == 'linear':
                    vals = normalize_linear(vals, min_category[col], max_category[col])
                norm_category[:,col] = vals
            setattr(self, '%s_norm_%s' % (norm, category_name), norm_category)

    def compute_parameter_importances(self, target='features', param_norm='dynamic', target_norm=None, num_trees=50,
                                      tree_height=25, uniform=False, n_neighbors=60, seed=None, plot=False):
        """
        Estimate the importance of each parameter in predicting the value of each target feature or objective.
        Uses an average of a random forest of decision tree regressors.
        :param target: str in ['features', 'objectives']
        :param param_norm: None or str in ['linear', 'dynamic']
        :param target_norm: None or str in ['linear', 'dynamic']
        :param num_trees: int
        :param tree_height: int
        :param uniform: reduce sampling bias by choosing points that are uniformly distributed, given non-uniform data
        :param n_neighbors:
        :param seed: int
        :param plot: bool
        :return: 2d array of float
        """
        from sklearn.ensemble import ExtraTreesRegressor
        from diversipy import psa_select

        # mtry = max(1, int(.1 * len(input_names)))
        num_params = self.parameters.shape[1]
        if target == 'features':
            num_targets = self.features.shape[1]
            Y_labels = self.feature_names
        elif target == 'objectives':
            num_targets = self.objectives.shape[1]
            Y_labels = self.feature_names
        else:
            raise Exception('SensitivityAnalysis.compute_parameter_importances: target must be features or objectives, '
                            'not %s' % target)
        param_importances = np.zeros((num_params, num_targets))

        if param_norm is None:
            X = self.parameters
        elif param_norm == 'dynamic':
            if self.dynamic_norm_parameters is None:
                self.normalize('parameters', norm='dynamic')
            X = self.dynamic_norm_parameters
        elif param_norm == 'linear':
            if self.linear_norm_parameters is None:
                self.normalize('parameters', norm='linear')
            X = self.linear_norm_parameters
        else:
            raise Exception('SensitivityAnalysis.compute_parameter_importances: param_norm must be linear, dynamic, or '
                            'None')

        if target_norm is None:
            Y = getattr(self, target)
        elif target_norm == 'dynamic':
            Y = getattr(self, 'dynamic_norm_%s' % target)
            if Y is None:
                self.normalize(target, norm='dynamic')
            Y = getattr(self, 'dynamic_norm_%s' % target)
        elif target_norm == 'linear':
            Y = getattr(self, 'linear_norm_%s' % target)
            if Y is None:
                self.normalize(target, norm='linear')
            Y = getattr(self, 'linear_norm_%s' % target)
        else:
            raise Exception('SensitivityAnalysis.compute_parameter_importances: target_norm must be linear, dynamic, or '
                            'None')

        # create a forest for each target
        for i in range(num_targets):
            rf = ExtraTreesRegressor(random_state=seed, max_depth=tree_height, n_estimators=num_trees)  # max_features=mtry,
            y = Y[:, i]
            if uniform and np.min(y) != np.max(y):
                y = Y[:, i].reshape(-1, 1)
                y_subset = psa_select(y, n_neighbors)
                _, idx, _ = np.intersect1d(y, y_subset, return_indices=True)
                X_subset = X[idx, :]
                y_subset = y[idx, 0]
                rf.fit(X_subset, y_subset)
            else:

                rf.fit(X, y)

            param_importances[:, i] = rf.feature_importances_

        if plot:

            fig, ax = plt.subplots()
            im = ax.imshow(param_importances, aspect='auto', interpolation='none')
            cbar = plt.colorbar(im, ax=ax)
            ax.set_xticks(np.arange(len(Y_labels)), labels=Y_labels)
            ax.set_yticks(np.arange(len(self.param_names)), labels=self.param_names)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

            # Loop over data dimensions and create text annotations.
            for i in range(len(Y_labels)):
                for j in range(len(self.param_names)):
                    text = ax.text(i, j, '%.2f' % param_importances[j, i], ha="center", va="center", color="w")

            ax.set_title('Gini parameter importances')
            fig.tight_layout()
            fig.show()

        return param_importances


def pop_to_matrix(history, input_str, output_str, param_strings, obj_strings):
    """converts collection of individuals in OptimizationHistory into a matrix for data manipulation

    :param history: OptimizationHistory object
    :return: data: 2d array. rows = each data point or individual, col = parameters, then features
    """
    total_models = np.sum([len(x) for x in history.generations])
    if total_models == 0:
        return np.array([]), np.array([])
    if input_str in param_strings:
        X_data = np.zeros((total_models, len(history.param_names)))
    elif input_str in obj_strings:
        X_data = np.zeros((total_models, len(history.objective_names)))
    else:
        X_data = np.zeros((total_models, len(history.feature_names)))
    y_data = np.zeros((total_models, len(history.objective_names))) if output_str in obj_strings else \
        np.zeros((total_models, len(history.feature_names)))
    counter = 0
    for generation in history.generations:
        for datum in generation:
            y_data[counter] = datum.objectives if output_str in obj_strings else datum.features
            if input_str in param_strings:
                X_data[counter] = datum.x
            elif input_str in obj_strings:
                X_data[counter] = datum.objectives
            else:
                X_data[counter] = datum.features
            counter += 1
    return X_data, y_data


def outline_colormap(ax, outline_list, fill=False):
    """
    :param ax: pyplot axis
    :param outline_list: 2d list. nested lists have the idxs of the input variables that were confounds
    :return:
    """
    patch_list = []
    for o, inp_list in enumerate(outline_list):
        for inp in inp_list:
            new_patch = Rectangle((o, inp), 1, 1, fill=fill, edgecolor='blue', lw=1.5) #idx from bottom left
            ax.add_patch(new_patch)
            patch_list.append(new_patch)
    return patch_list


def output_text(text, txt_file, verbose):
    if verbose: print(text)
    if txt_file is not None:
        txt_file.write(text)
        txt_file.write("\n")


def write_settings_to_file(input_str, output_str, x0_str, indep_norm, dep_norm, global_log_indep, global_log_dep, beta,
                           rel_start, confound_baseline, p_baseline, repeat, txt_file):
    txt_file.write("***************************************************\n")
    txt_file.write("Independent variable: %s\n" %input_str)
    txt_file.write("Dependent variable: %s\n" % output_str)
    txt_file.write("x0: %s\n" % x0_str)
    txt_file.write("Beta: %.2f\n" % beta)
    txt_file.write("Alpha: %.2f\n" % rel_start)
    txt_file.write("Repeats?: %s\n" % repeat)
    txt_file.write("Confound baseline: %.2f\n" % confound_baseline)
    txt_file.write("P-value threshold: %.2f\n" % p_baseline)
    txt_file.write("Independent variable normalization: %s\n" % indep_norm)
    if indep_norm == 'loglin':
        txt = 'global' if global_log_indep else 'local'
        txt_file.write("Independent variable log normalization: %s\n" % txt)
    txt_file.write("Dependent variable normalization: %s\n" % dep_norm)
    if dep_norm == 'loglin':
        txt = 'global' if global_log_dep else 'local'
        txt_file.write("Dependent variable log normalization: %s\n" % txt)
    txt_file.write("***************************************************\n")

def check_name_valid(name):
    for invalid_ch in "\/:*?\"<>|":
        name = name.replace(invalid_ch, "-'")
    return name

#------------------lsa plot

def get_coef_and_plot(neighbor_matrix, X_normed, y_normed, input_names, y_names, save, save_format='png', plot=True):
    """compute coefficients between parameter and feature based on linear regression. also get p-val
    coef will always refer to the R coefficient linear regression between param X and feature y

    :param neighbor_matrix: 2d array of lists which contain neighbor indices
    :param X_normed: 2d array of input vars normalized
    :param y_normed: 2d array of output vars normalized
    :return:
    """
    num_input, num_output = neighbor_matrix.shape[0], neighbor_matrix.shape[1]
    coef_matrix = np.zeros((num_input, num_output))
    pval_matrix = np.ones((num_input, num_output))

    for inp in range(num_input):
        for out in range(num_output):
            neighbor_array = neighbor_matrix[inp][out]
            # (if np.array) != (if list) in Python, hence conditions look
            # a little un-Pythonic
            if neighbor_array is not None and len(neighbor_array):
                selection = list(neighbor_array)
                X_sub = X_normed[selection, inp]

                coef_matrix[inp][out] = abs(linregress(X_sub, y_normed[selection, out])[2])
                pval_matrix[inp][out] = linregress(X_sub, y_normed[selection, out])[3]
                if plot:
                    plot_neighbors(X_normed[:, inp], y_normed[:, out], neighbor_array,
                                   input_names[inp], y_names[out], "Final pass", save, save_format)
    return coef_matrix, pval_matrix

def create_failed_search_matrix(neighbor_matrix, n_neighbors, lsa_heatmap_values):
    """
    failure = not enough neighbors or confounded
    for each significant feature/parameter relationship identified, check if possible confounds are significant
    """
    failed_matrix = np.zeros_like(neighbor_matrix, dtype=float)

    # not enough neighbors
    for param in range(neighbor_matrix.shape[0]):
        for feat in range(neighbor_matrix.shape[1]):
            if neighbor_matrix[param][feat] is None \
                    or len(neighbor_matrix[param][feat]) < n_neighbors:
                failed_matrix[param][feat] = lsa_heatmap_values['no_neighbors']
    return failed_matrix

# adapted from https://stackoverflow.com/questions/42976693/python-pick-event-for-pcolor-get-pandas-column-and-index-value
class InteractivePlot(object):
    def __init__(self, plot_obj, searched, sa_obj=None, coef_matrix=None, pval_matrix=None, p_baseline=.05,
                 r_ceiling_val=None):
        """
        click-based colormap
        :param plot_obj: SensitivityPlots object
        :param searched: bool, whether sensitivity analysis has been conducted for the whole grid
        :param sa_obj: ParameterSensitivity object
        :param coef_matrix: 2d array of abs R coefficients
        :param pval_matrix: 2d array of p-value coefficients
        :param p_baseline: float between 0 and 1; alpha for significance
        :param r_ceiling_val: float between 0 and 1; vmax for plotting. can be set if desired (e.g. if
            one value in the plot has a large R coef and you don't want it to dominate the color scheme)
        """
        import matplotlib as mpl

        self.plot_obj = plot_obj
        self.sa_obj = sa_obj
        self.searched = searched
        # only relevant if searched is False
        # k = input index, v = list of output indices
        self.subset_searched = {}

        self.coef_matrix = plot_obj.coef_matrix if coef_matrix is None else coef_matrix
        self.pval_matrix = plot_obj.pval_matrix if pval_matrix is None else pval_matrix
        self.input_names = plot_obj.input_names
        self.y_names = plot_obj.y_names
        self.sig_confounds = plot_obj.failed_matrix
        self.p_baseline = p_baseline
        self.r_ceiling_val = r_ceiling_val
        self.data = None
        self.ax = None
        norm = mpl.colors.Normalize(vmin=0., vmax=.8)
        self.val_to_color = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.GnBu)

        self.plot(p_baseline, r_ceiling_val)

    def plot(self, p_baseline=.05, r_ceiling_val=None):
        fig, ax = plt.subplots(figsize=(16, 5))
        plt.title("Absolute R Coefficients", y=1.11)
        vmax = min(.7, max(.1, np.max(self.coef_matrix))) if r_ceiling_val is None else r_ceiling_val

        cmap = plt.cm.GnBu
        cmap.set_under((0, 0, 0, 0))
        data = np.where(self.pval_matrix > p_baseline, 0, self.coef_matrix)
        data = np.where(self.sig_confounds != 0, 0, data)
        self.data = data
        self.ax = ax
        ax.pcolor(data, cmap=cmap, vmin=.01, vmax=vmax, picker=1)
        annotate(data, vmax)
        cmap = plt.cm.Greys
        cmap.set_under((0, 0, 0, 0))
        if self.sig_confounds is not None:
            ax.pcolor(self.sig_confounds, cmap=cmap, vmin=.01, vmax=1)
        set_centered_axes_labels(ax, self.input_names, self.y_names)
        plt.xticks(rotation=-90)
        plt.yticks(rotation=0)
        create_custom_legend(ax)
        fig.canvas.mpl_connect('pick_event', self.onpick)
        plt.show()
        time.sleep(4)

    def onpick(self, event):
        x, y = np.unravel_index(event.ind, self.pval_matrix.shape)
        x, y = x[0], y[0] # idx from bottom left
        plot_dict = {self.input_names[x] : [self.y_names[y]]}
        outline = [[] for _ in range(len(self.y_names))]
        outline[y] = [x]

        # patch = outline_colormap(self.ax, outline, fill=True)[0]
        # plt.pause(0.001)
        # plt.draw()
        # patch.remove()
        if not self.searched and self.sa_obj:
            if x not in self.subset_searched or y not in self.subset_searched[x]:
                outline_colormap(self.ax, outline, fill=False)
                plt.draw()
                first_pass_neighbors, neighbors, confounds, coef, pval = self.sa_obj.single_pair_analysis(
                    x, y, self.plot_obj.query_neighbors[x])
                self.plot_obj.query_neighbors[x] = first_pass_neighbors
                self.plot_obj.confound_matrix[x][y] = confounds
                self.plot_obj.neighbor_matrix[x][y] = neighbors
                self._set_cell(self.ax, x, y, neighbors, coef, pval)
                if coef:
                    print("Absolute R-coefficient was %.3f with a p-value of %.3f." % (coef, pval))
                if x not in self.subset_searched:
                    self.subset_searched[x] = [y]
                else:
                    self.subset_searched[x].append(y)

        self.plot_obj.plot_scatter_plots(plot_dict=plot_dict, save=False, show=True, plot_confounds=True)

    def _set_cell(self, ax, input_idx, output_idx, neighbors, coef, pval):
        if len(neighbors) < self.sa_obj.n_neighbors:
            color = 'grey'
        elif pval > self.sa_obj.p_baseline:
            color = 'white'
        else:
            color = self.val_to_color.to_rgba(coef)
        new_patch = Rectangle((output_idx, input_idx), 1, 1, facecolor=color)
        ax.add_patch(new_patch)
        plt.draw()
        if color != 'white' and color != 'grey':
            _, vmax = self.val_to_color.get_clim()
            txt_color = 'black' if vmax - coef > .45 * vmax else 'white'
            ax.text(output_idx + .5, input_idx + .5, '%.3f' % coef, ha='center',
                    va='center', color=txt_color)


def set_centered_axes_labels(ax, input_names, y_names):
    ax.set_yticks(np.arange(len(input_names)) + 0.5, minor=False)
    ax.set_xticks(np.arange(len(y_names)) + 0.5, minor=False)
    ax.set_yticklabels(input_names, minor=False)
    ax.set_xticklabels(y_names, minor=False)

#apparently matplotlib doesn't have a way to do this automatically like seaborn..
def annotate(data, vmax):
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            if data[y, x] == 0: continue
            if np.isnan(vmax):
                vmax = np.max(data[:, x])
            color = 'black' if vmax - data[y, x] > .45 * vmax else 'white'
            plt.text(x + 0.5, y + 0.5, '%.3f' % data[y, x], ha='center', va='center',
                     color=color)

#from https://stackoverflow.com/questions/49223702/adding-a-legend-to-a-matplotlib-plot-with-a-multicolored-line
class HandlerColorLineCollection(HandlerLineCollection):
    def create_artists(self, legend, artist ,xdescent, ydescent, width, height, fontsize, trans):
        x = np.linspace(0,width,self.get_numpoints(legend)+1)
        y = np.zeros(self.get_numpoints(legend)+1)+height/2.-ydescent
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=artist.cmap, transform=trans)
        lc.set_array(x)
        lc.set_linewidth(artist.get_linewidth())
        return [lc]

def create_custom_legend(ax, colormap='GnBu'):
    nonsig = plt.Line2D((0, 1), (0, 0), color='white', marker='s', mec='k',
                        mew=.5, linestyle='')
    no_neighbors = plt.Line2D((0, 1), (0, 0), color='#f3f3f3', marker='s',
                              linestyle='')
    sig = LineCollection(np.zeros((2, 2, 2)), cmap=colormap, linewidth=5)
    labels = ["Not significant",  "Too few neighbors",  "Significant without confounds"]
    ax.legend([nonsig, no_neighbors, sig], labels,
              handler_map={sig: HandlerColorLineCollection(numpoints=4)},
              loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=4, fancybox=True, shadow=True)

#------------------

class SobolPlot(object):
    def __init__(self, total, total_conf, first_order, first_order_conf, second_order, second_order_conf, input_names,
                 y_names, err_bars, title="Total effects"):
        """
        if plotting outside of the optimize pipeline:
          from nested.optimize_utils import *
          history = OptimizationHistory(hdf5_file_path)
          sobol_analysis(config_path, history)
        if in jupyter:
          sp = sobol_analysis(config_path, history, jupyter=True, feat=True)  # feat=False for objectives
          sp.plot_interactive()
        :param total: 2d array
        :param total_conf: 2d array
        :param first_order: 2d array
        :param first_order_conf: 2d array
        :param second_order: dict. key = str (independent var name), val = 2d array
        :param input_names: list
        :param y_names: list
        """
        self.total = total
        self.total_conf = total_conf
        self.first_order = first_order
        self.first_order_conf = first_order_conf
        self.second_order = second_order
        self.second_order_conf = second_order_conf
        self.y_names = y_names
        self.input_names = input_names
        self.ax = None
        self.acceptable_columns = None
        self.err_bars = err_bars
        self.title = title

        self.plot()

    def plot(self):
        fig, ax = plt.subplots()
        plt.title(self.title)
        # if a feature/objective does not vary at all, the column is a row of NaNs, which messes up
        # onpick event clicking
        self.acceptable_columns = [x for x in range(self.total.shape[1]) if not np.isnan(self.total[:, x]).any()]
        self.ax = ax
        ax.pcolor(self.total[:, self.acceptable_columns], cmap='GnBu', picker=1)
        self.annotate(
            self.total[:, self.acceptable_columns],
            self.total_conf[:, self.acceptable_columns], np.max(self.total))
        set_centered_axes_labels(
            ax, self.input_names, np.array(self.y_names)[self.acceptable_columns])
        plt.xticks(rotation=-90)
        plt.yticks(rotation=0)
        fig.canvas.mpl_connect('pick_event', self.onpick)
        plt.show()
        plt.close()

    def plot_interactive(self):
        """jupyter only"""
        import ipywidgets as widgets
        plot_types = ['First', 'Second', 'First and second']
        out = widgets.Dropdown(
            options=self.y_names,
            value=self.y_names[0],
            description="DV:",
            disabled=False,
        )

        plot_type = widgets.ToggleButtons(
            options=plot_types,
            description='Order:',
            disabled=False,
            button_style='',
        )

        err_bars = widgets.Checkbox(True, description='Error bars?')
        widgets.interact(
            self._interactive_helper, output_name=out, plot_type=plot_type, err_bars=err_bars)

    def _interactive_helper(self, output_name, plot_type, err_bars):
        output_idx = np.where(np.array(self.y_names) == output_name)[0][0]
        if plot_type.find("First") != -1:
            self._plot_first_order_effects(output_idx, err_bars=err_bars)
        if plot_type.lower().find("second") != -1:
            self._plot_second_order_effects(output_idx)

    def _plot_second_order_effects(self, output_idx):
        fig, ax = plt.subplots()
        plt.title("Second-order effects on {}".format(self.y_names[output_idx]))

        data = self.second_order[self.y_names[output_idx]]
        conf = self.second_order_conf[self.y_names[output_idx]]
        ax.pcolor(data, cmap='GnBu')
        self.annotate(data, conf, np.max(data))
        set_centered_axes_labels(ax, self.input_names, self.input_names)
        plt.xticks(rotation=-90)
        plt.yticks(rotation=0)

    def _plot_first_order_effects(self, output_idx, err_bars=True):
        fig, ax = plt.subplots()
        width = .35
        total_err = self.total_conf[:, output_idx] if err_bars else np.zeros((len(self.input_names),))
        first_err = self.first_order_conf[:, output_idx] if err_bars else np.zeros((len(self.input_names),))
        capsize = 2. if err_bars else 0.

        rect1 = ax.bar(np.arange(len(self.input_names)), self.total[:, output_idx],
                       width, align='center', yerr=total_err, label="Total effects",
                       capsize=capsize)
        rect2 = ax.bar(np.arange(len(self.input_names)) + width,
                       self.first_order[:, output_idx], width, align='center',
                       yerr=first_err, label="First order effects", capsize=capsize)
        plt.legend()
        autolabel(rect1, ax)
        autolabel(rect2, ax)
        plt.title("First order vs total effects on {}".format(self.y_names[output_idx]))
        plt.xticks(np.arange(len(self.input_names)) + width / 2, self.input_names,
                   rotation=-90)
        plt.ylabel('Effect')
        plt.yticks(rotation=0)

    def onpick(self, event):
        _, y = np.unravel_index(event.ind, (self.total.shape[0], len(self.acceptable_columns)))
        y = y[0] # idx from bottom left
        outline = [[] for _ in range(len(self.y_names))]
        outline[y] = [i for i in range((len(self.input_names)))]

        patch_list = outline_colormap(self.ax, outline, fill=True)
        plt.pause(0.001)
        plt.draw()
        for patch in patch_list:
            patch.remove()
        self._plot_first_order_effects(self.acceptable_columns[y], self.err_bars)
        self._plot_second_order_effects(self.acceptable_columns[y])
        plt.show()

    def annotate(self, arr, conf, vmax):
        # skip if confidence interval contains 0
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                if arr[r, c] - conf[r, c] <= 0 <= arr[r, c] + conf[r, c]:
                    continue
                if np.isnan(vmax):
                    vmax = np.max(arr[:, c])
                color = 'black' if vmax - arr[r, c] > .45 * vmax else 'white'
                plt.text(c + 0.5, r + 0.5, '%.3f' % arr[r, c], ha='center', va='center',
                         color=color)


# https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


#------------------plot importance via ensemble



#------------------user input

def prompt(accept, message, give_hint=False):
    user_input = ''
    while user_input not in accept:
        if give_hint:
            print('Valid input strings are: %s.' % accept)
        user_input = input(message + ': ').lower()
    return user_input

def get_variable_names(history, input_str, output_str, obj_strings, feat_strings, param_strings):
    if input_str in obj_strings:
        input_names = history.objective_names
    elif input_str in feat_strings:
        input_names = history.feature_names
    elif input_str in param_strings:
        input_names = history.param_names
    else:
        raise RuntimeError('SA: input variable "%s" is not recognized' % input_str)

    if output_str in obj_strings:
        y_names = history.objective_names
    elif output_str in feat_strings:
        y_names = history.feature_names
    elif output_str in param_strings:
        raise RuntimeError(
            "SA: parameters are currently not an acceptable dependent variable.")
    else:
        raise RuntimeError(
            'SA: output variable "%s" is not recognized' % output_str)
    return np.array(input_names), np.array(y_names)


def check_save_format_correct(save_format):
    accepted = ['png', 'pdf', 'svg']
    if save_format not in accepted:
        raise RuntimeError("For the save format for the plots, %s is not an "
                           "accepted string. Accepted strings are: %s." %
                           (save_format, accepted))


#------------------explore vector

def denormalize(scaling, unnormed_vector, param, logdiff_array, logmin_array, diff_array, min_array):
    if scaling[param] == 'log':
        unnormed_vector = np.power(
            10, (unnormed_vector * logdiff_array[param] + logmin_array[param]))
    else:
        unnormed_vector = unnormed_vector * diff_array[param] + min_array[param]

    return unnormed_vector

def get_param_bounds(config_file_path):
    """
    :param config_file_path: str
    :return: 2d array of shape (d, 2) where n is the number of parameters
    """
    from nested.utils import read_from_yaml
    yaml_dict = read_from_yaml(config_file_path)
    bounds_dict = yaml_dict['bounds']
    params = yaml_dict['param_names']
    bounds = np.zeros((len(bounds_dict), 2))
    for i, name in enumerate(params):
        try:
            bounds[i] = np.array(bounds_dict[name])
        except KeyError:
            raise RuntimeError("The parameter %s does not have specified bounds in the config file." % name)
    return bounds

def check_parameter_bounds(bounds, center, width, param_name):
    oob = False
    if center - width < bounds[0]:
        print("For the parameter %s, the perturbation vector includes values "
              "outside the lower bound of %.2f." % (param_name, bounds[0]))
        oob = True
    if center + width > bounds[1]:
        print("For the parameter %s, the perturbation vector includes values "
              "outside the upper bound of %.2f." % (param_name, bounds[1]))
        oob = True
    return oob

#------------------




def get_var_idx(var_name, var_dict):
    try:
        idx = var_dict[var_name]
    except KeyError:
        raise RuntimeError('The provided variable name %s is incorrect. Valid choices '
                           'are: %s.' % (var_name, list(var_dict.keys())))
    return idx

def get_var_idx_agnostic(var_name, input_dict, output_dict):
    if var_name not in input_dict.keys() and var_name not in output_dict.keys():
        raise RuntimeError('The provided variable name %s is incorrect. Valid choices '
                           'are: %s.' % (var_name, list(input_dict.keys()) + list(output_dict.keys())))
    elif var_name in input_dict.keys():
        return input_dict[var_name], True
    elif var_name in output_dict.keys():
        return output_dict[var_name], False


def sum_objectives(history, n):
    summed_obj = np.zeros((n,))
    counter = 0
    for generation in history.generations:
        for datum in generation:
            if datum.objectives is None:
                summed_obj[counter] = np.NaN
            else:
                summed_obj[counter] = sum(abs(datum.objectives))
            counter += 1
    return summed_obj


def convert_user_query_dict(queries, input_names, y_names):
    """converts user-supplied string values to indices, with error-checking"""
    idxs = defaultdict(list)
    incorrect_vals = []
    for inp_name, outputs in queries.items():
        if inp_name not in input_names:
            incorrect_vals.append(inp_name)
        for output_name in outputs:
            if output_name.lower() != 'all' and output_name not in y_names:
                incorrect_vals.append(output_name)
            elif inp_name in input_names:
                if output_name.lower() == 'all':
                    idxs[np.where(input_names == inp_name)[0][0]] = list(range(len(input_names)))
                else:
                    idxs[np.where(input_names == inp_name)[0][0]].append(
                        np.where(y_names == output_name)[0][0])

    if incorrect_vals:
        raise RuntimeError(
            "Dictionary is incorrect. The key must be a string (independent variable) "
            "and the value a list of strings (dependent variables). Incorrect strings "
            "were: %s. " % incorrect_vals)
    return idxs


def plot_r_hm(pval_matrix, coef_matrix, input_names, output_names, p_baseline=.05):
    fig, ax = plt.subplots()
    mask = np.full_like(pval_matrix, True, dtype=bool)
    mask[pval_matrix < p_baseline] = False

    # edge case where the shape is (n, )
    if len(pval_matrix.shape) < 2:
        coef_matrix = coef_matrix.reshape(1, -1)
        mask = mask.reshape(1, -1)
    hm = sns.heatmap(coef_matrix, mask=mask, cmap='cool', fmt=".2f",
                     linewidths=1, ax=ax, cbar=True, annot=True)
    hm.set_xticklabels(output_names)
    hm.set_yticklabels(input_names)
    plt.xticks(rotation=-90)
    plt.yticks(rotation=0)
    plt.show()