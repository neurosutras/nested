from nested.optimize_utils import PopulationStorage, Individual, OptimizationReport
import h5py
import numpy as np
import seaborn as sns
from collections import defaultdict
from scipy.stats import linregress, iqr
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import warnings
import time
from sklearn.ensemble import ExtraTreesRegressor
from matplotlib.backends.backend_pdf import PdfPages
import io

def local_sensitivity(population=None, X=None, y=None, input_names=None, y_names=None, x0_idx=None, x0_string=None,
                      input_str=None, output_str=None, no_lsa=False, indep_norm=None, dep_norm=None, n_neighbors=60,
                      max_neighbors=np.inf, beta=2., rel_start=.5, p_baseline=.05, confound_baseline=.5, r_ceiling_val=None,
                      important_dict=None, global_log_indep=None, global_log_dep=None, verbose=True, repeat=False, save=True,
                      save_path='data/lsa', save_format='png', save_txt=True):
    """
    the main function to run sensitivity analysis. provide either
        1) a PopulationStorage object
        2) the independent and dependent variables as two separate arrays

    :param population: PopulationStorage object.
    :param X: 2d np array or None (default). columns = variables, rows = examples
    :param y: 2d np array or None (default). columns = variables, rows = examples
    :param input_names: 1d np array or None (default). if PopulationStorage is given, no need to specify input_names.
        if X is given, the default names are 'input 0,' 'input 1'... etc unless input_names is given
    :param y_names: 1d np array or None (default). if PopulationStorage is given, no need to specify y_names.
        if y is given, the default names are 'output 0,' 'output 1'... etc unless y_names is given
    :param x0_idx: int or None (default). index of the center in the X array/PopulationStorage object
    :param x0_string: string or None. specify either x0_idx or x0_string, but not both. if both are None, a random
        center is selected. x0_string represents the center point of the neighbor search. accepted strings are 'best' or
        any of the objective names
    :param input_str: string representing the independent variable. accepted strings are 'parameter', 'p,' objective,'
        'o,' 'feature,' 'f.'
    :param output_str: string representing the independent variable. accepted strings are 'objective,'
        'o,' 'feature,' 'f.'
    :param no_lsa: bool; if true, sensitivity analysis is not done, but the LSA object is returned. this allows for
        convenient unfiltered plotting of the optimization.
    :param indep_norm: string specifying how the dependent variable is normalized. 'loglin,' 'lin', or 'none' are
        accepted strings.
    :param dep_norm: string specifying how the independent variable is normalized. 'loglin,' 'lin', or 'none' are
        accepted strings.
    :param n_neighbors: int. The minimum amount of neighbors desired to be selected during the first pass.
    :param max_neighbors: int or None. The maximum amount of neighbors desired to be selected during the first pass.
        If None, no maximum.
    :param beta: float. represents the maximum distance a nonquery parameter can vary relative to the query parameter
        during the first pass, i.e., a scalar factor.
    :param rel_start: float. represents the maximum distance a nonquery confound parameter can vary relative to the query
        parameter during clean-up. if repeat is True, the relative allowed distance is gradually decremented until there
        are no more confounds.
    :param p_baseline: float between 0 and 1. Threshold for statistical significance.
    :param confound_baseline: float between 0 and 1. Threshold for the absolute R coefficient a variable needs in
        order to be considered a confound.
    :param r_ceiling_val: float between 0 and 1, or None. If specified, all the colormaps in first_pass_colormaps.pdf
        will have a maximum of r_ceiling_val. This is to standardize the plots.
    :param important_dict: Dictionary. The keys are strings (dependent variable names) and the values are lists of strings
        (independent variable names). The user can specify what s/he already knows are important relationships.
    :param global_log_indep: string or None. if indep_norm is 'loglin,' user can specify if normalization should be
        global or local. accepted strings are 'local' or 'global.'
    :param global_log_dep: string or None. if dep_norm is 'loglin,' user can specify if normalization should be
        global or local. accepted strings are 'local' or 'global.'
    :param verbose: Bool; if true, prints out which variables were confounds in the set of points. Once can also
        see the confounds in first_pass_colormaps.pdf if save is True.
    :param repeat: Bool; if true, repeatededly checks the set of points to see if there are still confounds.
    :param save: Bool; if true, all neighbor search plots are saved.
    :param save_path: String that specifies the save path for the .hdf5 file containing the mildly perturbed
        independent variables as well as the neighbor search plots if save is true.
    :param save_format: string: 'png,' 'pdf,' or 'svg.' 'png' is the default. this specifies how the scatter plots
        will be saved (if they are saved)
    :param save_txt: bool; if True, will save the printed output in a text file in save_path
    :return: PopulationStorage and LSA object. The PopulationStorage contains the perturbations. The LSA object is
        for plotting and saving results of the optimization and/or sensitivity analysis.
    """
    #static
    feat_strings = ['f', 'feature', 'features']
    obj_strings = ['o', 'objective', 'objectives']
    param_strings = ['parameter', 'p', 'parameters']
    lsa_heatmap_values = {'confound': .35, 'no_neighbors': .1}

    check_save_format_correct(save_format)
    check_data_format_correct(population, X, y)
    #prompt user
    if x0_string is None and population is not None: x0_string = prompt_indiv(list(population.objective_names))
    if input_str is None and population is not None: input_str = prompt_input()
    if output_str is None and population is not None:  output_str = prompt_output()
    if indep_norm is None: indep_norm = prompt_norm("independent")
    if dep_norm is None: dep_norm = prompt_norm("dependent")

    if indep_norm == 'loglin' and global_log_indep is None: global_log_indep = prompt_global_vs_linear("n independent")
    if dep_norm == 'loglin' and global_log_dep is None: global_log_dep = prompt_global_vs_linear(" dependent")

    #set variables based on user input
    if input_names is None and y_names is None:
        if population is None:
            input_names = np.array(["input " + str(i) for i in range(X.shape[1])])
            y_names = np.array(["output " + str(i) for i in range(y.shape[1])])
        else:
            input_names, y_names = get_variable_names(population, input_str, output_str, obj_strings, feat_strings,
                                                      param_strings)

    if important_dict is not None: check_user_importance_dict_correct(important_dict, input_names, y_names)
    txt_file = io.open("{}/{}{}{}{}{}{}_output_txt.txt".format(save_path, *time.localtime()), "w", encoding='utf-8') \
               if save_txt else None
    if txt_file is not None:
        write_settings_to_file(
            input_str, output_str, x0_string, indep_norm, dep_norm, global_log_indep, global_log_dep, beta, rel_start,
            confound_baseline, p_baseline, repeat, txt_file)
    inp_out_same = (input_str in feat_strings and output_str in feat_strings) or \
                   (input_str in obj_strings and output_str in obj_strings)

    #process and potentially normalize data
    if population is not None: X, y = pop_to_matrix(population, input_str, output_str, param_strings, obj_strings)
    if x0_idx is not None:
        if population is not None:
            x0_idx = x0_to_index(population, x0_string, X, input_str, param_strings, obj_strings)
        elif x0_idx is not None:
            x0_idx = np.random.randint(0, X.shape[1])

    processed_data_X, crossing_X, z_X, pure_neg_X = process_data(X)
    processed_data_y, crossing_y, z_y, pure_neg_y = process_data(y)
    X_normed, scaling, logdiff_array, logmin_array, diff_array, min_array = normalize_data(
        processed_data_X, crossing_X, z_X, pure_neg_X, input_names, indep_norm, global_log_indep)
    y_normed, _, _, _, _, _ = normalize_data(
        processed_data_y, crossing_y, z_y, pure_neg_y, y_names, dep_norm, global_log_dep)
    if dep_norm is not 'none' and indep_norm is not 'none': print("Data normalized.")
    X_x0_normed = X_normed[x0_idx]

    if no_lsa:
        lsa_obj = LSA(pop=population, input_id2name=input_names, y_id2name=y_names, X=X_normed, y=y_normed, x0_idx=x0_idx,
                      processed_data_y=processed_data_y, crossing_y=crossing_y, z_y=z_y, pure_neg_y=pure_neg_y,
                      lsa_heatmap_values=lsa_heatmap_values)
        print("No exploration vector generated.")
        return None, lsa_obj

    plot_gini(X_normed, y_normed, X.shape[1], y.shape[1], input_names, y_names, inp_out_same)
    neighbors_per_query = first_pass(
        X_normed, y_normed, input_names, y_names, max_neighbors, beta, x0_idx, save_path, save, save_format, txt_file)
    neighbor_matrix, confound_matrix = clean_up(neighbors_per_query, X_normed, y_normed, X_x0_normed, input_names, y_names,
                                                n_neighbors, r_ceiling_val, save_path, p_baseline, confound_baseline,
                                                rel_start, repeat, save, save_format, txt_file, verbose)

    lsa_obj = LSA(pop=population, neighbor_matrix=neighbor_matrix, query_neighbors=neighbors_per_query,
                  input_id2name=input_names, y_id2name=y_names, X=X_normed, y=y_normed, x0_idx=x0_idx,
                  processed_data_y=processed_data_y, crossing_y=crossing_y, z_y=z_y, pure_neg_y=pure_neg_y,
                  n_neighbors=n_neighbors, confound_matrix=confound_matrix, lsa_heatmap_values=lsa_heatmap_values)

    coef_matrix, pval_matrix = interactive_colormap(
        lsa_obj, dep_norm, global_log_dep, processed_data_y, crossing_y, z_y, pure_neg_y, neighbor_matrix, X_normed,
        y_normed, input_names, y_names, n_neighbors, lsa_heatmap_values, p_baseline, r_ceiling_val, save_path,
        save, save_format)

    lsa_obj.coef_matrix = coef_matrix
    lsa_obj.pval_matrix = pval_matrix
    if txt_file is not None: txt_file.close()

    if input_str not in param_strings and population is not None:
        explore_pop = None
        print("The exploration vector for the parameters was not generated because it was not the dependent variable.")
    else:
        explore_dict = generate_explore_vector(n_neighbors, X.shape[1], y.shape[1], X[x0_idx], X_x0_normed,
                                               scaling, logdiff_array, logmin_array, diff_array, min_array,
                                               neighbor_matrix, indep_norm)
        if population is None:
            explore_pop = convert_dict_to_PopulationStorage(explore_dict, input_names, y_names, y_names, save_path)
        else:
            explore_pop = convert_dict_to_PopulationStorage(explore_dict, population.param_names, population.feature_names,
                                                            population.objective_names, save_path)

    return explore_pop, lsa_obj


def interactive_colormap(lsa_obj, dep_norm, global_log_dep, processed_data_y, crossing_y, z_y, pure_neg_y, neighbor_matrix,
                         X_normed, y_normed, input_names, y_names, n_neighbors, lsa_heatmap_values, p_baseline,
                         r_ceiling_val, save_path, save, save_format):
    old_dep_norm = None
    old_global_dep = None
    num_input = X_normed.shape[1]
    num_output = y_normed.shape[1]
    plot = True
    while plot:
        if old_dep_norm != dep_norm or old_global_dep != global_log_dep:
            y_normed, _, _, _, _, _ = normalize_data(
                processed_data_y, crossing_y, z_y, pure_neg_y, y_names, dep_norm, global_log_dep)
            coef_matrix, pval_matrix = get_coef_and_plot(
                num_input, num_output, neighbor_matrix, X_normed, y_normed, input_names, y_names, save_path, save, save_format)
        failed_matrix = create_failed_search_matrix(
            num_input, num_output, neighbor_matrix, n_neighbors, lsa_heatmap_values)
        plotSensitivity(lsa_obj, coef_matrix, pval_matrix, input_names, y_names, failed_matrix, p_baseline, r_ceiling_val)
        old_dep_norm = dep_norm
        old_global_dep = global_log_dep
        p_baseline, r_ceiling_val, dep_norm, global_log_dep, plot = prompt_plotting(
            p_baseline, r_ceiling_val, dep_norm, global_log_dep)
    return coef_matrix, pval_matrix

#------------------processing populationstorage and normalizing data

def pop_to_matrix(population, input_str, output_str, param_strings, obj_strings):
    """converts collection of individuals in PopulationStorage into a matrix for data manipulation

    :param population: PopulationStorage object
    :return: data: 2d array. rows = each data point or individual, col = parameters, then features
    """
    X_data = []
    y_data = []
    generation_array = population.history
    for generation in generation_array:
        for datum in generation:
            y_array = datum.objectives if output_str in obj_strings else datum.features
            y_data.append(y_array)
            if input_str in param_strings:
                x_array = datum.x
            elif input_str in obj_strings:
                x_array = datum.objectives
            else:
                x_array = datum.features
            X_data.append(x_array)
    return np.array(X_data), np.array(y_data)


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

    return processed_data, crossing, z, pure_neg


def x0_to_index(population, x0_string, X_data, input_str, param_strings, obj_strings):
    """
    from x0 string (e.g. 'best'), returns the respective array/data which contains
    both the parameter and output values
    """
    report = OptimizationReport(population)
    if x0_string == 'best':
        if input_str in param_strings:
            x0_x_array = report.survivors[0].x
        elif input_str in obj_strings:
            x0_x_array = report.survivors[0].objectives
        else:
            x0_x_array = report.survivors[0].features
    else:
        if input_str in param_strings:
            x0_x_array = report.specialists[x0_string].x
        elif input_str in obj_strings:
            x0_x_array = report.specialists[x0_string].objectives
        else:
            x0_x_array = report.specialists[x0_string].features
    index = np.where(X_data == x0_x_array)[0][0]
    return index


def normalize_data(processed_data, crossing, z, pure_neg, names, norm, global_log=None, magnitude_threshold=2):
    """normalize all data points. used for calculating neighborship

    :param processed_data: data has been transformed for the cols that need to be log-normalized such that the values
        can be logged
    :param crossing: list of column indices such that within the column, values cross 0
    :param z: list of column idx such that column has a 0
    :return: matrix of normalized values for parameters and features
    """
    # process_data DOES NOT process the columns (ie, parameters and features) that cross 0, because
    # that col will just be lin normed.
    warnings.simplefilter("ignore")

    data_normed = np.copy(processed_data)
    num_rows, num_cols = processed_data.shape

    min_array, diff_array = get_linear_arrays(processed_data)
    diff_array[np.where(diff_array == 0)[0]] = 1
    data_log_10 = np.log10(np.copy(processed_data))
    logmin_array, logdiff_array, logmax_array = get_log_arrays(data_log_10)

    scaling = []  # holds a list of whether the column was log or lin normalized (string)
    if norm == 'loglin':
        scaling = np.array(['log'] * num_cols)
        if global_log is True:
            scaling[np.where(logdiff_array < magnitude_threshold)[0]] = 'lin'
        else:
            n = logdiff_array.shape[0]
            scaling[np.where(logdiff_array[-int(n / 3):] < magnitude_threshold)[0]] = 'lin'
        scaling[crossing] = 'lin'; scaling[z] = 'lin'
        lin_loc = np.where(scaling == 'lin')[0]
        log_loc = np.where(scaling == 'log')[0]
        print("Normalization: %s." % list(zip(names, scaling)))
    elif norm == 'lin':
        scaling = np.array(['lin'] * num_cols)
        lin_loc = range(num_cols)
        log_loc = []
    else:
        lin_loc = []
        log_loc = []

    data_normed[:, lin_loc] = np.true_divide((processed_data[:, lin_loc] - min_array[lin_loc]), diff_array[lin_loc])
    data_normed[:, log_loc] = np.true_divide((data_log_10[:, log_loc] - logmin_array[log_loc]),
                                             logdiff_array[log_loc])
    data_normed = np.nan_to_num(data_normed)
    data_normed[:, pure_neg] *= -1

    return data_normed, scaling, logdiff_array, logmin_array, diff_array, min_array

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

#------------------independent variable importance

def add_user_knowledge(user_important_dict, y_name, imp):
    if user_important_dict is not None and y_name in user_important_dict.keys():
        for known_imp_input in user_important_dict[y_name]:
            if known_imp_input not in imp:
                imp.append(known_imp_input)

def accept_outliers(coef):
    IQR = iqr(coef)
    q3 = np.percentile(coef, 75)
    upper_baseline = q3 + 1.5 * IQR
    return set(np.where(coef > upper_baseline)[0])

#------------------consider all variables unimportant at first

def first_pass(X, y, input_names, y_names, max_neighbors, beta, x0_idx, save_path, save, save_format, txt_file):
    neighbor_arr = [[] for _ in range(X.shape[1])]
    x0_normed = X[x0_idx]
    X_dists = np.abs(X - x0_normed)
    output_text(
        "First pass: ",
        txt_file,
        True,
    )
    for i in range(X.shape[1]):
        neighbors = []
        unimp = [x for x in range(X.shape[1]) if x != i]
        X_dists_sorted = X_dists[X_dists[:, i].argsort()]

        for j in range(X.shape[0]):
            X_dist_sub = X_dists_sorted[j]
            rad = X_dists_sorted[j][i]
            if np.all(np.abs(X_dist_sub[unimp]) <= beta * rad):
                idx = np.where(X_dists == X_dists_sorted[j])[0][0]
                neighbors.append(idx)
            if len(neighbors) >= max_neighbors: break
        neighbor_arr[i] = neighbors
        max_dist = np.max(X_dists[neighbors][:, i])
        output_text(
            "    %s - %d neighbors found. Max query distance of %.8f. " % (input_names[i], len(neighbors), max_dist),
            txt_file,
            True,
        )
        for o in range(y.shape[1]):
            plot_neighbors(X[neighbors][:, i], y[neighbors][:, o], input_names[i], y_names[o], "First pass", save_path,
                           save, save_format)

    return neighbor_arr

def clean_up(neighbor_arr, X, y, X_x0, input_names, y_names, n_neighbors, r_ceiling_val, save_path, p_baseline,
             confound_baseline, rel_start, repeat, save, save_format, txt_file, verbose):
    num_input = len(neighbor_arr)
    neighbor_matrix = np.empty((num_input, y.shape[1]), dtype=object)
    confound_matrix = np.empty((num_input, y.shape[1]), dtype=object)
    pdf = PdfPages("%s/first_pass_colormaps.pdf" % save_path) if save else None
    for i in range(num_input):
        nq = [x for x in range(num_input) if x != i]
        neighbor_orig = neighbor_arr[i].copy()
        confound_list = [[] for _ in range(y.shape[1])]
        for o in range(y.shape[1]):
            neighbors = neighbor_arr[i].copy()
            iter = 0
            current_confounds = None
            rel = rel_start
            while current_confounds is None or (rel > 0 and len(current_confounds) != 0 and len(neighbors) > n_neighbors):
                current_confounds = []
                rmv_list = []
                for i2 in nq:
                    r = abs(linregress(X[neighbors][:, i2], y[neighbors][:, o])[2])
                    pval = linregress(X[neighbors][:, i2], y[neighbors][:, o])[3]
                    if r >= confound_baseline and pval < p_baseline:
                        output_text(
                            "Iteration %d: For the set of neighbors associated with %s vs %s, %s was significantly "
                                "correlated with %s." % (iter, input_names[i], y_names[o], input_names[i2], y_names[o]),
                            txt_file,
                            verbose,
                        )
                        current_confounds.append(i2)
                        plot_neighbors(X[neighbors][:, i2], y[neighbors][:, o], input_names[i2], y_names[o],
                                       "Clean up (query parameter = %s)" % (input_names[i]), save_path, save, save_format)
                        for n in neighbors:
                            if abs(X[n, i2] - X_x0[i2]) > rel * abs(X[n, i] - X_x0[i]):
                                if n not in rmv_list: rmv_list.append(n)
                for n in rmv_list: neighbors.remove(n)
                output_text(
                    "During iteration %d, for the pair %s vs %s, %d points were removed. %d remain." \
                        % (iter, input_names[i], y_names[o], len(rmv_list), len(neighbors)),
                    txt_file,
                    verbose,
                )
                if iter == 0:
                    confound_matrix[i][o] = current_confounds
                    confound_list[o] = current_confounds
                if not repeat: break
                rel -= (rel_start / 10.)
                iter += 1
            neighbor_matrix[i][o] = neighbors if not repeat or (repeat and len(current_confounds) == 0) else []
            plot_neighbors(X[neighbors][:, i], y[neighbors][:, o], input_names[i], y_names[o], "Final pass",
                           save_path, save, save_format)
            if len(neighbors) < n_neighbors:
                output_text(
                    "----Clean up: %s vs %s - %d neighbor(s) remaining!" % (input_names[i], y_names[o], len(neighbors)),
                    txt_file,
                    True,
                )
        plot_first_pass_colormap(neighbor_orig, X, y, input_names, y_names, input_names[i], confound_list, p_baseline,
                                 r_ceiling_val, pdf, save)
    if save: pdf.close()
    return neighbor_matrix, confound_matrix


def plot_neighbors(a, b, input_name, y_name, title, save_path, save, save_format, close=True):
    plt.figure()
    plt.scatter(a, b)
    plt.ylabel(y_name)
    plt.xlabel(input_name)
    plt.title(title)
    if len(a) > 1:
        r = abs(linregress(a, b)[2])
        pval = linregress(a, b)[3]
        fit_fn = np.poly1d(np.polyfit(a, b, 1))
        plt.plot(a, fit_fn(a), color='red')
        plt.title("{} - Abs R = {:.2e}, p-val = {:.2e}".format(title, r, pval))
    if save: plt.savefig('%s/%s_%s_vs_%s.%s' % (save_path, title, input_name, y_name, save_format), format=save_format)
    if close: plt.close()


def plot_first_pass_colormap(neighbors, X, y, input_names, y_names, input_name, confound_list, p_baseline=.05, r_ceiling_val=None,
                             pdf=None, save=True):
    coef_matrix = np.zeros((X.shape[1], y.shape[1]))
    pval_matrix = np.zeros((X.shape[1], y.shape[1]))
    for i in range(X.shape[1]):
        for o in range(y.shape[1]):
            coef_matrix[i][o] = abs(linregress(X[neighbors, i], y[neighbors, o])[2])
            pval_matrix[i][o] = linregress(X[neighbors, i], y[neighbors, o])[3]
    fig, ax = plt.subplots(figsize=(16, 5))
    plt.title("Absolute R Coefficients - First pass of %s" % input_name)
    vmax = np.max(coef_matrix) if r_ceiling_val is None else r_ceiling_val
    cmap = plt.cm.GnBu
    cmap.set_under((0, 0, 0, 0))
    coef_matrix_masked = np.where(pval_matrix > p_baseline, 0, coef_matrix)
    ax.pcolor(coef_matrix_masked, cmap=cmap, vmin=0.01, vmax=vmax)
    annotate(coef_matrix_masked, vmax)
    set_centered_axes_labels(ax, input_names, y_names)
    outline_colormap(ax, confound_list)
    plt.xticks(rotation=-90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save: pdf.savefig(fig)
    plt.close()


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

def output_text(str, txt_file, verbose):
    if verbose: print(str)
    if txt_file is not None:
        txt_file.write(str)
        txt_file.write(u"\r\n")

def write_settings_to_file(input_str, output_str, x0_str, indep_norm, dep_norm, global_log_indep, global_log_dep, beta,
                           rel_start, confound_baseline, p_baseline, repeat, txt_file):
    txt_file.write("***************************************************" + u"\r\n")
    txt_file.write("Independent variable: %s" %input_str + u"\r\n")
    txt_file.write("Dependent variable: %s" % output_str + u"\r\n")
    txt_file.write("x0: %s" % x0_str + u"\r\n" )
    txt_file.write("Beta: %.2f" % beta + u"\r\n" )
    txt_file.write("Alpha: %.2f" % rel_start + u"\r\n")
    txt_file.write("Repeats?: %s" % repeat + u"\r\n" )
    txt_file.write("Confound baseline: %.2f" % confound_baseline + u"\r\n" )
    txt_file.write("P-value threshold: %.2f" % p_baseline + u"\r\n" )
    txt_file.write("Independent variable normalization: %s" % indep_norm + u"\r\n")
    if indep_norm == 'loglin':
        str = 'global' if global_log_indep else 'local'
        txt_file.write("Independent variable log normalization: %s" % str + u"\r\n" )
    txt_file.write("Dependent variable normalization: %s" % dep_norm + u"\r\n" )
    if dep_norm == 'loglin':
        str = 'global' if global_log_dep else 'local'
        txt_file.write("Dependent variable log normalization: %s" % str + u"\r\n")
    txt_file.write("***************************************************" + u"\r\n")


#------------------lsa plot

def get_coef_and_plot(num_input, num_output, neighbor_matrix, X_normed, y_normed, input_names, y_names, save_path, save,
                      save_format):
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
            if neighbor_array is not None and len(neighbor_array) > 0:
                selection = list(neighbor_array)
                X_sub = X_normed[selection, inp]  # get relevant X data points

                coef_matrix[inp][out] = abs(linregress(X_sub, y_normed[selection, out])[2])
                pval_matrix[inp][out] = linregress(X_sub, y_normed[selection, out])[3]
                plot_neighbors(X_normed[neighbor_array, inp], y_normed[neighbor_array, out], input_names[inp], y_names[out],
                               "Final pass",  save_path, save, save_format)
    return coef_matrix, pval_matrix

def create_failed_search_matrix(num_input, num_output, neighbor_matrix, n_neighbors, lsa_heatmap_values):
    """
    failure = not enough neighbors or confounded
    for each significant feature/parameter relationship identified, check if possible confounds are significant
    """
    failed_matrix = np.zeros((num_input, num_output))

    # not enough neighbors
    for param in range(num_input):
        for feat in range(num_output):
            if neighbor_matrix[param][feat] is None or len(neighbor_matrix[param][feat]) < n_neighbors:
                failed_matrix[param][feat] = lsa_heatmap_values['no_neighbors']
    return failed_matrix


# adapted from https://stackoverflow.com/questions/42976693/python-pick-event-for-pcolor-get-pandas-column-and-index-value
class plotSensitivity(object):
    def __init__(self, lsa_obj, coef_matrix, pval_matrix, input_names, y_names, sig_confounds, p_baseline=.05,
                 r_ceiling_val=None):
        self.lsa_obj = lsa_obj
        self.coef_matrix = coef_matrix
        self.pval_matrix = pval_matrix
        self.input_names = input_names
        self.y_names = y_names
        self.sig_confounds = sig_confounds
        self.p_baseline = p_baseline
        self.r_ceiling_val = r_ceiling_val
        self.data = None
        self.ax = None
        self.plot(coef_matrix, pval_matrix, input_names, y_names, sig_confounds, p_baseline, r_ceiling_val)

    def plot(self, coef_matrix, pval_matrix, input_names, y_names, sig_confounds, p_baseline=.05, r_ceiling_val=None):
        fig, ax = plt.subplots(figsize=(16, 5))
        plt.title("Absolute R Coefficients", y=1.11)
        vmax = min(.7, max(.1, np.max(coef_matrix))) if r_ceiling_val is None else r_ceiling_val

        cmap = plt.cm.GnBu
        cmap.set_under((0, 0, 0, 0))
        data = np.where(pval_matrix > p_baseline, 0, coef_matrix)
        data = np.where(sig_confounds != 0, 0, data)
        self.data = data
        self.ax = ax
        ax.pcolor(data, cmap=cmap, vmin=0.01, vmax=vmax, picker=1)
        annotate(data, vmax)
        cmap = plt.cm.Greys
        cmap.set_under((0, 0, 0, 0))
        ax.pcolor(sig_confounds, cmap=cmap, vmin=.01, vmax=1)
        set_centered_axes_labels(ax, input_names, y_names)
        plt.xticks(rotation=-90)
        plt.yticks(rotation=0)
        create_LSA_custom_legend(ax)
        fig.canvas.mpl_connect('pick_event', self.onpick)
        plt.show()
        plt.close()

    def onpick(self, event):
        x, y = np.unravel_index(event.ind, self.pval_matrix.shape)
        x, y = x[0], y[0] # idx from bottom left
        plot_dict = {self.input_names[x] : [self.y_names[y]]}
        outline = [[] for _ in range(len(self.y_names))]
        outline[y] = [x]

        patch = outline_colormap(self.ax, outline, fill=True)[0]
        plt.pause(0.001)
        plt.draw()
        patch.remove()
        self.lsa_obj.first_pass_scatter_plots(plot_dict=plot_dict, save=False, close=False)
        self.lsa_obj.clean_up_scatter_plots(plot_dict=plot_dict, save=False, close=False)
        plt.show()

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
            color = 'black' if vmax - data[y, x] > .45 * vmax else 'white'
            plt.text(x + 0.5, y + 0.5, '%.3f' % data[y, x], ha='center', va='center', color=color)

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

def create_LSA_custom_legend(ax, colormap='GnBu'):
    nonsig = plt.Line2D((0, 1), (0, 0), color='white', marker='s', mec='k', mew=.5, linestyle='')
    no_neighbors = plt.Line2D((0, 1), (0, 0), color='#f3f3f3', marker='s', linestyle='')
    sig = LineCollection(np.zeros((2, 2, 2)), cmap=colormap, linewidth=5)
    labels = ["Not significant",  "Too few neighbors",  "Significant without confounds"]
    ax.legend([nonsig, no_neighbors, sig], labels,
              handler_map={sig: HandlerColorLineCollection(numpoints=4)}, loc='upper center',
              bbox_to_anchor=(0.5, 1.12), ncol=4, fancybox=True, shadow=True)

#------------------plot importance via ensemble

def plot_gini(X, y, num_input, num_output, input_names, y_names, inp_out_same):
    num_trees = 50
    tree_height = 25
    mtry = max(1, int(.1 * len(input_names)))
    # the sum of feature_importances_ is 1, so the baseline should be relative to num_input
    # the below calculation is pretty ad hoc and based fitting on (20, .1), (200, .05), (2000, .01); (num_input, baseline)
    baseline = 0.15688 - 0.0195433 * np.log(num_input)
    if baseline < 0: baseline = .005
    #important_inputs = [[] for _ in range(num_output)]
    input_importances = np.zeros((num_input, num_output))

    # create a forest for each feature. each independent var is considered "important" if over the baseline
    for i in range(num_output):
        rf = ExtraTreesRegressor(random_state=0, max_features=mtry, max_depth=tree_height, n_estimators=num_trees)
        Xi = X[:, [x for x in range(num_input) if x != i]] if inp_out_same else X
        rf.fit(Xi, y[:, i])

        # imp_loc = list(set(np.where(rf.feature_importances_ >= baseline)[0]) | accept_outliers(rf.feature_importances_))
        feat_imp = rf.feature_importances_
        if inp_out_same:
            # imp_loc = [x + 1 if x >= i else x for x in imp_loc]
            feat_imp = np.insert(feat_imp, i, np.NaN)
        input_importances[:, i] = feat_imp
        # important_inputs[i] = list(input_names[imp_loc])

    fig, ax = plt.subplots()
    hm = sns.heatmap(input_importances, cmap='cool', fmt=".2f", linewidths=1, ax=ax, cbar=True, annot=True)
    hm.set_xticklabels(y_names)
    hm.set_yticklabels(input_names)
    plt.xticks(rotation=-90)
    plt.yticks(rotation=0)
    plt.title('Gini importances')
    plt.show()

#------------------user input prompts

def prompt_plotting(alpha, r_ceiling, y_norm, global_y_norm):
    user_input = ''
    while user_input.lower() not in ['y', 'yes', 'n', 'no']:
        user_input = input('Do you want to replot the figure with new plotting parameters (alpha value, '
                           'R ceiling, etc)?: ')
    if user_input.lower() in ['y', 'yes']:
        y_norm, global_norm = prompt_change_y_norm(y_norm)
        return prompt_alpha(), prompt_r_ceiling_val(), y_norm, global_norm, True
    else:
        return alpha, r_ceiling, y_norm, global_y_norm, False

def prompt_alpha():
    alpha = ''
    while alpha is not float:
        try:
            alpha = input('Alpha value? Default is 0.05: ')
            return float(alpha)
        except ValueError:
            print('Please enter a float.')
    return .05

def prompt_r_ceiling_val():
    r_ceiling_val = ''
    while r_ceiling_val is not float:
        try:
            r_ceiling_val = input('What should the ceiling for the absolute R value be in the plot?: ')
            return float(r_ceiling_val)
        except ValueError:
            print('Please enter a float.')
    return .7

def prompt_confound_baseline():
    baseline = ''
    while baseline is not float:
        try:
            baseline = input('What should the minimum absolute R coefficient of a variable be for it to be considered '
                             'a confound? The default is 0.5: ')
            return float(baseline)
        except ValueError:
            print('Please enter a float.')
    return .5

def prompt_change_y_norm(prev_norm):
    user_input = ''
    while user_input not in ['lin', 'global loglin', 'local loglin', 'none']:
        user_input = input('For plotting, do you want to change the way the dependent variable is normalized? Current '
                           'normalization is: %s. (Answers: lin/global loglin/local loglin/none) ' % prev_norm).lower()
    global_norm = None
    if user_input.find('loglin') != -1:
        if user_input.find('global') != -1: global_norm = True
        if user_input.find('local') != -1: global_norm = False
        user_input = 'loglin'
    return user_input, global_norm

def prompt_indiv(valid_names):
    user_input = ''
    while user_input != 'best' and user_input not in valid_names:
        print('Valid strings for x0: %s.' % (['best'] + valid_names))
        user_input = (input('Specify x0: ')).lower()

    return user_input

def prompt_norm(variable_string):
    user_input = ''
    while user_input.lower() not in ['lin', 'loglin', 'none']:
        user_input = input('How should %s variables be normalized? Accepted answers: lin/loglin/none: ' % variable_string)
    return user_input.lower()

def prompt_global_vs_linear(variable_str):
    user_input = ''
    while user_input.lower() not in ['g', 'global', 'l', 'local']:
        user_input = input('For determining whether a%s variable is log normalized, should its value across all '
                           'generations be examined or only the last third? Accepted answers: local/global: '
                           % variable_str)
    return user_input.lower() in ['g', 'global']

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
    return np.array(input_names), np.array(y_names)

def check_user_importance_dict_correct(dct, input_names, y_names):
    incorrect_strings = []
    for y_name in dct.keys():
        if y_name not in y_names: incorrect_strings.append(y_name)
    for _, known_important_inputs in dct.items():
        if not isinstance(known_important_inputs, list):
            raise RuntimeError('For the known important variables dictionary, the value must be a list, even if '
                               'the list contains only one variable.')
        for name in known_important_inputs:
            if name not in input_names: incorrect_strings.append(name)
    if len(incorrect_strings) > 0:
        raise RuntimeError('Some strings in the dictionary are incorrect. Are the keys dependent variables (string) '
                           'and the values independent variables (list of strings)? These inputs have errors: %s.'
                           % incorrect_strings)

def check_save_format_correct(save_format):
    accepted = ['png', 'pdf', 'svg']
    if save_format not in accepted:
        raise RuntimeError("For the save format for the plots, %s is not an accepted string. Accepted strings are: "
                           "%s." % (save_format, accepted))

def check_data_format_correct(population, X, y):
    if (population is not None and (X is not None or y is not None)) \
            or (population is None and (X is None or y is None)):
        raise RuntimeError("Please either give one PopulationStorage object or a pair of arrays.")

#------------------explore vector

def denormalize(scaling, unnormed_vector, param, logdiff_array, logmin_array, diff_array, min_array):
    if scaling[param] == 'log':
        unnormed_vector = np.power(10, (unnormed_vector * logdiff_array[param] + logmin_array[param]))
    else:
        unnormed_vector = unnormed_vector * diff_array[param] + min_array[param]

    return unnormed_vector

def create_perturb_matrix(X_x0, n_neighbors, input, perturbations):
    """
    :param X_best: x0
    :param n_neighbors: int, how many perturbations were made
    :param input: int, idx for independent variable to manipulate
    :param perturbations: array
    :return:
    """
    perturb_matrix = np.tile(np.array(X_x0), (n_neighbors, 1))
    perturb_matrix[:, input] = perturbations
    return perturb_matrix

def generate_explore_vector(n_neighbors, num_input, num_output, X_x0, X_x0_normed, scaling, logdiff_array,
                            logmin_array, diff_array, min_array, neighbor_matrix, norm_search):
    """
    figure out which X/y pairs need to be explored: non-sig or no neighbors
    generate n_neighbor points around best point. perturb just POI... 5% each direction

    :return: dict, key=param number (int), value=list of arrays
    """
    explore_dict = {}
    if n_neighbors % 2 == 1: n_neighbors += 1

    for inp in range(num_input):
        for output in range(num_output):
            if neighbor_matrix[inp][output] is None or len(neighbor_matrix[inp][output]) < n_neighbors:
                upper = .05 * np.random.random_sample((int(n_neighbors / 2),)) + X_x0_normed[inp]
                lower = .05 * np.random.random_sample((int(n_neighbors / 2),)) + X_x0_normed[inp] - .05
                unnormed_vector = np.concatenate((upper, lower), axis=0)

                perturbations = unnormed_vector if norm_search is 'none' else denormalize(
                    scaling, unnormed_vector, inp, logdiff_array, logmin_array, diff_array, min_array)
                perturb_matrix = create_perturb_matrix(X_x0, n_neighbors, inp, perturbations)
                explore_dict[inp] = perturb_matrix
                break

    return explore_dict

def save_perturbation_PopStorage(perturb_dict, param_id2name, save_path=''):
    import time
    full_path = save_path + '/{}_{}_{}_{}_{}_{}_perturbations'.format(*time.localtime())
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

#------------------

class LSA(object):
    """"
    allows for plotting and re-plotting after sensitivity analysis has been conducted. can be saved or loaded. for example,
    lsa_plotter = LSA(file_path='path/to/lsa.pkl')
    """
    def __init__(self, pop=None, neighbor_matrix=None, coef_matrix=None, pval_matrix=None, query_neighbors=None,
                 confound_matrix=None, input_id2name=None, y_id2name=None, X=None, y=None, x0_idx=None, processed_data_y=None,
                 crossing_y=None, z_y=None, pure_neg_y=None, n_neighbors=None, lsa_heatmap_values=None):
        self.neighbor_matrix = neighbor_matrix
        self.query_neighbors = query_neighbors
        self.confound_matrix = confound_matrix
        self.coef_matrix = coef_matrix
        self.pval_matrix = pval_matrix
        self.X = X
        self.y = y
        self.x0_idx = x0_idx
        self.lsa_heatmap_values = lsa_heatmap_values
        self.summed_obj = sum_objectives(pop, X.shape[0]) if pop is not None else None

        self.processed_data_y = processed_data_y
        self.crossing_y = crossing_y
        self.z_y = z_y
        self.pure_neg_y = pure_neg_y
        self.n_neighbors = n_neighbors

        self.input_names = input_id2name
        self.y_names = y_id2name
        self.input_name2id = {}
        self.y_name2id = {}

        for i, name in enumerate(input_id2name): self.input_name2id[name] = i
        for i, name in enumerate(y_id2name): self.y_name2id[name] = i


    def plot_final_colormap(self, dep_norm='none', global_log_dep=None, r_ceiling_val=.7, p_baseline=.05):
        """
        plots the final colormap of absolute R values that one sees at the end of sensitivity analysis.

        :param dep_norm: string. specifies how the dependent variable will be normalized. default is 'none.' other accepted
           strings are 'loglin' and 'lin.'
        :param global_log_dep: string or None. if dep_norm is 'loglin,' then the user can specify if the normalization
           should be done globally or locally. accepted strings are 'local' and 'global.'
        :param
        """
        if self.neighbor_matrix is None:
            raise RuntimeError("LSA was not done.")
        interactive_colormap(self, dep_norm, global_log_dep, self.processed_data_y, self.crossing_y, self.z_y, self.pure_neg_y,
                             self.neighbor_matrix, self.X, self.y, self.input_names, self.y_names, self.n_neighbors,
                             self.lsa_heatmap_values, p_baseline, r_ceiling_val, save_path='data/lsa', save=False, save_format='png')

    def plot_vs_filtered(self, input_name, y_name, x_axis=None, y_axis=None):
        """
        plots the set of points associated with a specific independent/dependent variable pair.

        :param input_name: string. independent variable name.
        :param y_name: string. dependent variable name.
        :param x_axis: string or None. default is None. if None, the x axis of the plot is the same as input_name
        :param y_axis: string or None. default is None. if None, the y axis of the plot is the same as y_name
        :return:
        """
        if self.neighbor_matrix is None:
            raise RuntimeError("SA was not run. Please use plot_vs_unfiltered() instead.")
        if (x_axis is None or y_axis is None) and x_axis != y_axis:
            raise RuntimeError("Please specify both or none of the axes.")

        input_id = get_var_idx(input_name, self.input_name2id)
        output_id, = get_var_idx(y_name, self.y_name2id)
        if x_axis is not None:
            x_id, input_bool_x = get_var_idx_agnostic(x_axis, self.input_name2id, self.y_name2id)
            y_id, input_bool_y = get_var_idx_agnostic(y_axis, self.input_name2id, self.y_name2id)
        else:
            x_id = input_id
            y_id = output_id
            input_bool_x = True
            input_bool_y = False

        neighbor_indices = self.neighbor_matrix[input_id][output_id]
        if neighbor_indices is None or len(neighbor_indices) <= 1:
            print("No neighbors-- nothing to show.")
        else:
            a = self.X[neighbor_indices, x_id] if input_bool_x else self.y[neighbor_indices, x_id]
            b = self.X[neighbor_indices, y_id] if input_bool_y else self.y[neighbor_indices, y_id]
            plt.scatter(a, b)
            plt.scatter(self.X[self.x0_idx, x_id], self.y[self.x0_idx, y_id], color='red', marker='+')
            fit_fn = np.poly1d(np.polyfit(a, b, 1))
            plt.plot(a, fit_fn(a), color='red')

            plt.title("{} vs {} with p-val of {:.2e} and R coef of {:.2e}.".format(
                input_name, y_name, self.pval_matrix[input_id][output_id], self.coef_matrix[input_id][output_id]))

            plt.xlabel(input_name)
            plt.ylabel(y_name)
            plt.show()


    def plot_vs_unfiltered(self, x_axis, y_axis, num_models=None, last_third=False):
        """
        plots any two variables against each other. does not use the filtered set of points gathered during
        sensitivity analysis.

        :param x_axis: string. name of in/dependent variable
        :param y_axis: string. name of in/dependent variable
        :param num_models: int or None. if None, plot all models. else, plot the last num_models.
        :param last_third: bool. if True, use only the values associated with the last third of the optimization
        """
        x_id, input_bool_x = get_var_idx_agnostic(x_axis, self.input_name2id, self.y_name2id)
        y_id, input_bool_y = get_var_idx_agnostic(y_axis, self.input_name2id, self.y_name2id)

        if num_models is not None:
            num_models = int(num_models)
            x = self.X[-num_models:, x_id] if input_bool_x else self.y[-num_models:, x_id]
            y = self.X[-num_models:, y_id] if input_bool_y else self.y[-num_models:, y_id]
            plt.scatter(x, y, c=self.summed_obj[-num_models:], cmap='viridis_r')
            plt.title("Last {} models.".format(num_models))
        elif last_third:
            m = int(self.X.shape[0] / 3)
            x = self.X[-m:, x_id] if input_bool_x else self.y[-m:, x_id]
            y = self.X[-m:, y_id] if input_bool_y else self.y[-m:, y_id]
            plt.scatter(x, y, c=self.summed_obj[-m:], cmap='viridis_r')
            plt.title("Last third of models.")
        else:
            x = self.X[:, x_id] if input_bool_x else self.y[:, x_id]
            y = self.X[:, y_id] if input_bool_y else self.y[:, y_id]
            plt.scatter(x, y, c=self.summed_obj, cmap='viridis_r')
            plt.title("All models.")

        plt.scatter(self.X[self.x0_idx, x_id], self.y[self.x0_idx, y_id], color='red', marker='+')
        plt.colorbar().set_label("Summed objectives")
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.show()


    def first_pass_color_map(self, inputs=None, p_baseline=.05, r_ceiling_val=None, save=True, save_path='data/lsa'):
        """
        there is a unique set of points for each of the independent variables during the first pass. for each of the sets
         specified, the linear relationship between each independent and dependent variable will be plotted.

        :param inputs: a list of strings of input variable names, or None. if None, the colormap for each variable
            will be plotted.
        :param p_baseline: a float from 0 to 1. threshold for statistical significance
        :param r_ceiling_val: a float from 0 to 1, or None. if specified, all of the colormaps plotted will have the
            same upper bound
        :param save: bool
        :param save_path: string. default is 'data/lsa.'
        """
        if self.query_neighbors is None:
            raise RuntimeError("SA was not run.")
        pdf = PdfPages("%s/first_pass_colormaps.pdf" % save_path) if save else None

        if inputs is None:
            query = [x for x in range(len(self.input_names))]
        else:
            query = []
            for input in inputs:
                try:
                    query.append(np.where(self.input_names == input)[0][0])
                except:
                    raise RuntimeError("One of the inputs specified is not correct. Valid inputs are: %s." % self.input_names)
        for i in query:
            plot_first_pass_colormap(self.query_neighbors[i], self.X, self.y, self.input_names, self.y_names,
                                     self.input_names[i], p_baseline, r_ceiling_val, pdf, save)
        if save: pdf.close()


    def first_pass_scatter_plots(self, plot_dict=None, close=False, save=True, save_format='png', save_path='data/lsa'):
        """
        plots the scatter plots during the naive search.

        :param plot_dict: dict or None. the key is a string (independent variable) and the value is a list of strings (of
            dependent variables). if None, all of the plots are plotted
        :param close: bool. if True, the plot does not appear, but it may be saved if save is True
        :param save: bool
        :param save_format: string: 'png,' 'svg,' or 'pdf.'
        :param save_path: string. default is 'data/lsa.'
        """

        idxs_dict = defaultdict(list)
        if plot_dict is not None: idxs_dict = convert_user_query_dict(plot_dict, self.input_names, self.y_names)
        if plot_dict is None:
            for i in range(len(self.input_names)):
                idxs_dict[i] =  range(len(self.y_names))
        for i, output_list in idxs_dict.items():
            for o in output_list:
                neighbors = self.query_neighbors[i]
                plot_neighbors(self.X[neighbors][:, i], self.y[neighbors][:, o], self.input_names[i], self.y_names[o],
                               "First pass", save_path=save_path, save=save, save_format=save_format, close=close)


    def clean_up_scatter_plots(self, plot_dict=None, close=False, save=True, save_format='png', save_path='data/lsa'):
        """
        plots the relationships after the clean-up search. if there were confounds in the naive set of neighbors,
            the relationship between the confound and the dependent variable of interest are also plotted.

        :param plot_dict: dict or None. the key is a string (independent variable) and the value is a list of strings (of
            dependent variables). if None, all of the plots are plotted
        :param close: bool. if True, the plot does not appear, but it may be saved if save is True
        :param save: bool
        :param save_format: string: 'png,' 'svg,' or 'pdf.'
        :param save_path: string. default is 'data/lsa.'
        """
        idxs_dict = defaultdict(list)
        if self.confound_matrix is None:
            raise RuntimeError('SA was not run.')
        if plot_dict is not None: idxs_dict = convert_user_query_dict(plot_dict, self.input_names, self.y_names)
        if plot_dict is None:
            for i in range(len(self.input_names)):
                idxs_dict[i] = range(len(self.y_names))
        for i, output_list in idxs_dict.items():
            for o in output_list:
                neighbors = self.query_neighbors[i]
                confounds = self.confound_matrix[i][o]
                if confounds is None:
                    print("%s vs. %s was not confounded." % (self.input_names[i], self.y_names[o]))
                else:
                    for confound in confounds:
                        plot_neighbors(self.X[neighbors][:, confound], self.y[neighbors][:, o], self.input_names[confound],
                                       self.y_names[o], "Clean up (query parameter = %s)" % (self.input_names[i]),
                                       save_path, save, save_format, close)
                final_neighbors = self.neighbor_matrix[i][o]
                plot_neighbors(self.X[final_neighbors][:, i], self.y[final_neighbors][:, o], self.input_names[i],
                               self.y_names[o], "Final", save_path, save, save_format, close)


    def return_filtered_data(self, input_name, y_name):
        input_id = get_var_idx(input_name, self.input_name2id)
        y_id = get_var_idx(y_name, self.y_name2id)
        neighbor_indices = self.neighbor_matrix[input_id][y_id]
        if neighbor_indices is None or len(neighbor_indices) <= 1:
            raise RuntimeError("No neighbors were found for this pair.")
        return self.X[neighbor_indices], self.y[neighbor_indices]


def get_var_idx(var_name, var_dict):
    try:
        idx = var_dict[var_name]
    except:
        raise RuntimeError('The provided variable name %s is incorrect. Valid choices are: %s.'
                           % (var_name, list(var_dict.keys())))
    return idx

def get_var_idx_agnostic(var_name, input_dict, output_dict):
    if var_name not in input_dict.keys() and var_name not in output_dict.keys():
        raise RuntimeError('The provided variable name %s is incorrect. Valid choices are: %s.'
                           % (var_name, list(input_dict.keys()) + list(output_dict.keys())))
    elif var_name in input_dict.keys():
        return input_dict[var_name], True
    elif var_name in output_dict.keys():
        return output_dict[var_name], False

def sum_objectives(pop, n):
    summed_obj = np.zeros((n,))
    counter = 0
    generation_array = pop.history
    for generation in generation_array:
        for datum in generation:
            summed_obj[counter] = sum(abs(datum.objectives))
            counter += 1
    return summed_obj

def convert_user_query_dict(dct, input_names, y_names):
    res = defaultdict(list)
    incorrect_input = []
    for k, li in dct.items():
        valid = True
        if k not in input_names:
            incorrect_input.append(k)
            valid = False
        for output_name in li:
            if output_name.lower() != 'all' and output_name not in y_names:
                incorrect_input.append(output_name)
            elif valid:
                if output_name.lower == 'all':
                    res[np.where(input_names == k)[0][0]] = [x for x in range(len(input_names))]
                else:
                    res[np.where(input_names == k)[0][0]].append(np.where(y_names == output_name)[0][0])

    if len(incorrect_input) > 0:
        raise RuntimeError("Dictionary is incorrect. The key must be a string (independent variable) and the value a "
                           "list of strings (dependent variables). Incorrect inputs were: %s. " % incorrect_input)
    return res
