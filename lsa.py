from nested.optimize_utils import PopulationStorage, Individual, OptimizationReport
import h5py
import collections
import numpy as np
from collections import defaultdict
from sklearn.neighbors import BallTree
from sklearn.decomposition import PCA
from scipy.stats import linregress, rankdata, iqr
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.collections import LineCollection
import math
import warnings
import pickle
import os.path
from sklearn.ensemble import ExtraTreesRegressor
import time


def local_sensitivity(population, x0_string=None, input_str=None, output_str=None, no_lsa=None, indep_norm=None, dep_norm=None, n_neighbors=None, max_dist=None, unimp_ub=None,
                      p_baseline=.05, confound_baseline=.5, r_ceiling_val=None, important_dict=None, global_log_indep=None,
                      global_log_dep=None, sig_radius_factor=2., timeout=np.inf, annotated=True, verbose=True, save_path=''):
    #static
    feat_strings = ['f', 'feature', 'features']
    obj_strings = ['o', 'objective', 'objectives']
    param_strings = ['parameter', 'p', 'parameters']
    lsa_heatmap_values = {'confound': .35, 'no_neighbors': .1}

    #prompt user
    if x0_string is None: x0_string = prompt_indiv(list(population.objective_names))
    if input_str is None: input_str = prompt_input()
    if output_str is None: output_str = prompt_output()
    if indep_norm is None: indep_norm = prompt_norm("independent")
    if dep_norm is None: dep_norm = prompt_norm("dependent")

    if no_lsa is None: no_lsa = prompt_no_lsa()
    if indep_norm == 'loglin' and global_log_indep is None: global_log_indep = prompt_global_vs_linear("n independent")
    if dep_norm == 'loglin' and global_log_dep is None: global_log_dep = prompt_global_vs_linear(" dependent")
    if not no_lsa and n_neighbors is None and max_dist is None: n_neighbors, max_dist = prompt_values()
    if max_dist is None: max_dist = prompt_max_dist()
    if n_neighbors is None: n_neighbors = prompt_num_neighbors()

    #set variables based on user input
    input_names, y_names = get_variable_names(population, input_str, output_str, obj_strings, feat_strings,
                                              param_strings)
    if important_dict is not None: check_user_importance_dict_correct(important_dict, input_names, y_names)
    num_input = len(input_names)
    num_output = len(y_names)
    input_is_not_param = input_str not in param_strings
    inp_out_same = (input_str in feat_strings and output_str in feat_strings) or \
                   (input_str in obj_strings and output_str in obj_strings)

    #process and potentially normalize data
    X, y = pop_to_matrix(population, input_str, output_str, param_strings, obj_strings)
    x0_idx = x0_to_index(population, x0_string, X, input_str, param_strings, obj_strings)
    processed_data_X, crossing_X, z_X, pure_neg_X = process_data(X)
    processed_data_y, crossing_y, z_y, pure_neg_y = process_data(y)
    X_normed, scaling, logdiff_array, logmin_array, diff_array, min_array = normalize_data(
        processed_data_X, crossing_X, z_X, pure_neg_X, input_names, indep_norm, global_log_indep)
    y_normed, _, _, _, _, _ = normalize_data(
        processed_data_y, crossing_y, z_y, pure_neg_y, y_names, dep_norm, global_log_dep)
    if dep_norm is not 'none' and indep_norm is not 'none': print("Data normalized.")
    X_x0_normed = X_normed[x0_idx]

    plot_gini(X_normed, y_normed, num_input, num_output, input_names, y_names, inp_out_same)
    first_neighbor_arr= first_pass(X_normed, X_x0_normed, n_neighbors, unimp_ub)
    important_inputs = get_important_inputs(
        first_neighbor_arr, X_normed, y_normed, input_names, y_names, important_dict, confound_baseline, p_baseline)
    """
    def __init__(self, population, neighbor_matrix, coef_matrix, pval_matrix, fail_matrix,
                 input_names, y_names, X_normed, y_normed, x0_idx, important_inputs,
                 processed_data_y, crossing_y, z_y, pure_neg_y, n_neighbor,
                 confound_matrix, lsa_heatmap_values)
    """

    if no_lsa:
        lsa_obj = LSA(population, None, None, None, None, None, input_names, y_names, X_normed, y_normed, x0_idx,
                      important_inputs, processed_data_y, crossing_y, z_y, pure_neg_y, n_neighbors, None, lsa_heatmap_values)
        print("No exploration vector generated.")
        return None, lsa_obj, None

    neighbor_matrix, confound_matrix, debugger_matrix, radii_matrix, unimportant_range, important_range \
        = prompt_neighbor_dialog(num_input, num_output, important_inputs, input_names, y_names, X_normed,
                                 x0_idx, verbose, n_neighbors, max_dist, inp_out_same, sig_radius_factor, timeout)
    coef_matrix, pval_matrix, fail_matrix = interactive_plot(
        num_input, num_output, neighbor_matrix, X_normed, y_normed, processed_data_y, crossing_y, z_y, pure_neg_y,
        n_neighbors, important_inputs, input_names, y_names, confound_matrix, dep_norm, global_log_dep, radii_matrix,
        x0_idx, unimportant_range, important_range, max_dist, debugger_matrix, sig_radius_factor, lsa_heatmap_values,
        annotated, r_ceiling_val, p_baseline, confound_baseline, timeout, verbose)

    #create objects to return
    lsa_obj = LSA(population, neighbor_matrix, coef_matrix, pval_matrix, fail_matrix, input_names, y_names, X_normed,
                  y_normed, x0_idx, important_inputs, processed_data_y, crossing_y, z_y, pure_neg_y, n_neighbors,
                  confound_matrix, lsa_heatmap_values)
    debug = InterferencePlot(debugger_matrix, X_normed, y_normed, input_names, y_names, important_inputs, radii_matrix)
    if input_is_not_param:
        explore_pop = None
    else:
        explore_dict = generate_explore_vector(n_neighbors, num_input, num_output, X[x0_idx], X_x0_normed,
                                               scaling, logdiff_array, logmin_array, diff_array, min_array,
                                               neighbor_matrix, indep_norm)
        explore_pop = convert_dict_to_PopulationStorage(explore_dict, input_names, population.feature_names,
                                                        population.objective_names, save_path)
    if input_is_not_param:
        print("The exploration vector for the parameters was not generated because it was not the dependent variable.")
    return explore_pop, lsa_obj, debug


def interactive_plot(num_input, num_output, neighbor_matrix, X_normed, y_normed, processed_data_y, crossing_y, z_y,
                     pure_neg_y, n_neighbors, important_inputs, input_names, y_names, confound_matrix, dep_norm,
                     global_log_dep, radii_matrix, x0_idx, unimportant_range, important_range, max_dist, debugger_matrix,
                     sig_radius_factor, lsa_heatmap_values, annotated=True, r_ceiling_val=None, p_baseline=.05,
                     confound_baseline=.5, timeout=np.inf, verbose=True):
    redo = True
    while redo is True:
        """old_dep_norm = None
        old_global_dep = None
        plot = True
        while plot:
            if old_dep_norm != dep_norm or old_global_dep != global_log_dep:
                y_normed, _, _, _, _, _ = normalize_data(
                    processed_data_y, crossing_y, z_y, pure_neg_y, y_names, dep_norm, global_log_dep)
                coef_matrix, pval_matrix = get_coef(num_input, num_output, neighbor_matrix, X_normed, y_normed)
            fail_matrix, confound_dict = create_failed_search_matrix(
                num_input, num_output, coef_matrix, pval_matrix, confound_matrix, input_names, y_names, important_inputs,
                neighbor_matrix, n_neighbors, lsa_heatmap_values, p_baseline, confound_baseline)
            plot_sensitivity(num_input, num_output, coef_matrix, pval_matrix, input_names, y_names, fail_matrix,
                             p_baseline, r_ceiling_val, annotated)
            p_baseline, r_ceiling_val, annotated, confound_baseline, dep_norm, global_log_dep, plot = prompt_plotting(
                p_baseline, r_ceiling_val, annotated, confound_baseline, dep_norm, global_log_dep)"""
        confound_dict, coef_matrix, pval_matrix, fail_matrix = interactive_colormap(
            dep_norm, global_log_dep, processed_data_y, crossing_y, z_y, pure_neg_y, neighbor_matrix, X_normed, y_normed,
            confound_matrix, input_names, y_names, important_inputs, n_neighbors, lsa_heatmap_values, p_baseline,
            confound_baseline, r_ceiling_val, annotated)
        redo = prompt_redo_confounds() if len(confound_dict.keys()) else False
        if redo:
            redo_confounds(confound_dict, important_inputs, y_names, max_dist, num_input, n_neighbors, radii_matrix,
                   input_names, X_normed, X_normed[x0_idx], debugger_matrix, neighbor_matrix, confound_matrix, x0_idx,
                   unimportant_range, important_range, sig_radius_factor, timeout, verbose)
    return coef_matrix, pval_matrix, fail_matrix


def interactive_colormap(dep_norm, global_log_dep, processed_data_y, crossing_y, z_y, pure_neg_y, neighbor_matrix,
                         X_normed, y_normed, confound_matrix, input_names, y_names, important_inputs, n_neighbors,
                         lsa_heatmap_values, p_baseline, confound_baseline, r_ceiling_val, annotated):
    old_dep_norm = None
    old_global_dep = None
    num_input = X_normed.shape[1]
    num_output = y_normed.shape[1]
    plot = True
    while plot:
        if old_dep_norm != dep_norm or old_global_dep != global_log_dep:
            y_normed, _, _, _, _, _ = normalize_data(
                processed_data_y, crossing_y, z_y, pure_neg_y, y_names, dep_norm, global_log_dep)
            coef_matrix, pval_matrix = get_coef(num_input, num_output, neighbor_matrix, X_normed, y_normed)
        fail_matrix, confound_dict = create_failed_search_matrix(
            num_input, num_output, coef_matrix, pval_matrix, confound_matrix, input_names, y_names, important_inputs,
            neighbor_matrix, n_neighbors, lsa_heatmap_values, p_baseline, confound_baseline)
        plot_sensitivity(num_input, num_output, coef_matrix, pval_matrix, input_names, y_names, fail_matrix,
                         p_baseline, r_ceiling_val, annotated)

        old_dep_norm = dep_norm
        old_global_dep = global_log_dep
        p_baseline, r_ceiling_val, annotated, confound_baseline, dep_norm, global_log_dep, plot = prompt_plotting(
            p_baseline, r_ceiling_val, annotated, confound_baseline, dep_norm, global_log_dep)
    return confound_dict, coef_matrix, pval_matrix, fail_matrix

#------------------processing populationstorage and normalizing data

def pop_to_matrix(population, input_str, output_str, param_strings, obj_strings):
    """converts collection of individuals in PopulationStorage into a matrix for data manipulation

    :param population: PopulationStorage object
    :param feat_bool: True if we're doing LSA on features, False if on objectives
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

    :param population: PopulationStorage object
    :param data: 2d array object with data from generations
    :param processed_data: data has been transformed for the cols that need to be log-normalized such that the values
                           can be logged
    :param crossing: list of column indices such that within the column, values cross 0
    :param z: list of column idx such that column has a 0
    :param x0_string: user input string specifying x0
    :param param_names: names of parameters
    :param input_is_not_param: bool
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

def filter_Euclidean(X, X_x0, rad):
    unimportant_tree = BallTree(X, metric='euclidean')
    unimportant_neighbor_array = unimportant_tree.query_radius(X_x0.reshape(1, -1), r=rad)[0]
    return unimportant_neighbor_array

def first_pass(X, X_x0, n_neighbors, upper_bound=None):
    num_inp = X.shape[1]
    if upper_bound is None: upper_bound = .1 * num_inp
    unimp_rad_increment = .005 * num_inp
    unimp_rad_start = .005 * num_inp
    neighbor_arr = []
    rad = unimp_rad_start
    start = time.time()
    while len(neighbor_arr) < n_neighbors:
        if rad > upper_bound:
            print("First pass: Neighbors not found for specified n_neighbor threshold. Best attempt: %d in %.2f seconds."
                  % len(neighbor_arr), time.time() - start)
            break

        neighbor_arr = filter_Euclidean(X, X_x0, rad)
        if len(neighbor_arr) >= n_neighbors:
            print("\nFirst pass: %d neighbors found within a radius of %.2f in %.2f seconds."
                  % (len(neighbor_arr), rad, time.time() - start))
        rad += unimp_rad_increment
    return neighbor_arr

def get_important_inputs(neighbor_arr, X, y, input_names, y_names, user_important_dict, confound_baseline=.5, alpha=.05):
    neighbor_matrix = np.full((X.shape[1], y.shape[1]), set(neighbor_arr), dtype=set)
    coef_matrix, pval_matrix = get_coef(X.shape[1], y.shape[1], neighbor_matrix, X, y)
    important_inputs = [[] for _ in range(y.shape[1])]
    print("Calculating important dependent variables: ")
    for o in range(y.shape[1]):
        for i in range(X.shape[1]):
            if pval_matrix[i][o] < alpha and coef_matrix[i][o] >= confound_baseline:
                important_inputs[o].append(input_names[i])
        add_user_knowledge(user_important_dict, y_names[o], important_inputs[o])
        print("    %s - %s" % (y_names[o], important_inputs[o]))
    print("Done.")
    return important_inputs

#------------------neighbor search

def create_distance_trees(unimportant, important, X_normed, X_x0_normed, unimportant_rad, important_rad):
    """make two BallTrees to do distance querying"""
    # get first set of neighbors (filter by important params)
    # second element of the tree query is dtype, which is useless
    if important:
        important_cheb_tree = BallTree(X_normed[:, important], metric='chebyshev')
        important_neighbor_array = important_cheb_tree.query_radius(
            X_x0_normed[important].reshape(1, -1), r=important_rad)[0]
    else:
        important_neighbor_array = np.array([])

    # get second set (by unimprt parameters)
    if unimportant:
        unimportant_tree = BallTree(X_normed[:, unimportant], metric='euclidean')
        unimportant_neighbor_array = unimportant_tree.query_radius(
            X_x0_normed[unimportant].reshape(1, -1), r=unimportant_rad)[0]
    else:
        unimportant_neighbor_array = np.array([])

    return unimportant_neighbor_array, important_neighbor_array


def filter_neighbors(x_not, important, unimportant, X_normed, X_x0_normed, important_rad, unimportant_rad, i, o,
                     debug_matrix, sig_radius_factor=2.):
    """filter according to the radii constraints and if query parameter perturbation > twice the max perturbation
    of important parameters
    passed neighbors = passes all constraints
    filtered neighbors = neighbors that fit the important input variable distance constraint + the distance of
        the input variable of interest is more than twice that of the important variable constraint"""

    unimportant_neighbor_array, important_neighbor_array = create_distance_trees(
        unimportant, important, X_normed, X_x0_normed, unimportant_rad, important_rad)
    if len(unimportant_neighbor_array) > 1 and len(important_neighbor_array) > 1:
        sig_perturbation = abs(X_normed[important_neighbor_array, i] - X_x0_normed[i]) >= sig_radius_factor * important_rad
        sig_neighbors = important_neighbor_array[sig_perturbation].tolist() + [x_not]
        passed_neighbors = [idx for idx in sig_neighbors if idx in unimportant_neighbor_array]
    else:
        sig_neighbors = [x_not]
        passed_neighbors = [x_not]

    debug_matrix = update_debugger(
        debug_matrix, unimportant_neighbor_array, important_neighbor_array, sig_neighbors, passed_neighbors, i, o)
    return passed_neighbors, debug_matrix


def compute_neighbor_matrix(num_input, num_output, important_inputs, input_names, y_names, X_normed,
                            x_not, verbose, n_neighbors, max_dist, inp_out_same, sig_radius_factor=2., timeout=np.inf):
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
    # initialize
    neighbor_matrix = np.empty((num_input, num_output), dtype=object)
    important_range = (float('inf'), float('-inf'))  # first element = min, second = max
    unimportant_range = (float('inf'), float('-inf'))
    confound_matrix = np.empty((num_input, num_output), dtype=object)
    debugger_matrix = defaultdict(dd)
    radii_matrix = np.empty((num_input, num_output), dtype=object)

    X_x0_normed = X_normed[x_not]

    for p in range(num_input):  # row
        for o in range(num_output):  # col
            if inp_out_same and p == o: continue
            start = time.time()
            unimportant_range, important_range = search(
                p, o, max_dist, num_input, important_inputs[o], n_neighbors, radii_matrix, input_names, y_names,
                X_normed, X_x0_normed, debugger_matrix, neighbor_matrix, confound_matrix, x_not,
                unimportant_range, important_range, sig_radius_factor, timeout, verbose)
            print("--------------Took %.2f seconds" % (time.time() - start))
    print("Important independent variable radius range:", important_range, "/ Unimportant:", unimportant_range)
    return neighbor_matrix, confound_matrix, debugger_matrix, radii_matrix, unimportant_range, important_range


def search(p, o, max_dist, num_inputs, important_input, n_neighbors, radii_matrix, input_names, y_names,
           X_normed, X_x0_normed, debugger_matrix, neighbor_matrix, confound_matrix, x_not,
           unimportant_range, important_range, sig_radius_factor=2., timeout=np.inf, verbose=True):
    unimp_rad_increment = .05
    unimp_rad_start = .1
    unimp_upper_bound = [.67 * x + .67 + unimp_rad_start for x in range(1, 4)]
    imp_rad_threshold = [.1 * x - .05 + max_dist for x in range(1, 4)]
    imp_rad_cutoff = imp_rad_threshold[-1]

    important_rad = max_dist
    magnitude = int(math.log10(max_dist))
    start = time.time()

    # split important vs unimportant parameters
    unimportant, important = split_parameters(num_inputs, important_input, input_names, p)
    scale = max(1, len(unimportant)) / 20
    filtered_neighbors = []
    best = (0, 0, 0)
    while len(filtered_neighbors) < n_neighbors:
        unimportant_rad = unimp_rad_start * scale

        # break if most of the important parameter space is being searched
        if important_rad > imp_rad_cutoff or time.time() - start > timeout:
            radii_matrix[p][o] = (unimportant_rad, important_rad)
            neighbor_matrix[p][o] = filtered_neighbors
            print("\nInput: %s / Output: %s - Neighbors not found for specified n_neighbor threshold. Best "
                  "attempt: %d neighbor(s) with unimportant radius of %.2f and important radius of %.2f. %s"
                  % (input_names[p], y_names[o], best[0], best[1], best[2],
                     difficult_constraint(debugger_matrix[(p, o)], unimportant, important)))
            if time.time() - start > timeout: print("Timed out.")
            break

        filtered_neighbors, debugger_matrix = filter_neighbors(
            x_not, important, unimportant, X_normed, X_x0_normed, important_rad, unimportant_rad, p, o,
            debugger_matrix, sig_radius_factor)
        if len(filtered_neighbors) > best[0]: best = (len(filtered_neighbors), unimportant_rad, important_rad)

        # print statement, update ranges, check confounds
        if len(filtered_neighbors) >= n_neighbors:
            unimportant_range, important_range = housekeeping(
                neighbor_matrix, p, o, filtered_neighbors, verbose, input_names, y_names, unimportant_rad,
                important_rad, unimportant_range, important_range, confound_matrix, X_x0_normed, X_normed,
                important, unimportant, radii_matrix)

        # if not enough neighbors are found, increment unimportant_radius until enough neighbors found
        # OR the radius is greater than important_radius*ratio
        if important_rad < imp_rad_threshold[0]:
            upper_bound = unimp_upper_bound[0] * scale
        elif important_rad < imp_rad_threshold[1]:
            upper_bound = unimp_upper_bound[1] * scale
        else:
            upper_bound = unimp_upper_bound[2] * scale

        while len(filtered_neighbors) < n_neighbors and unimportant_rad < upper_bound:
            filtered_neighbors, debugger_matrix = filter_neighbors(
                x_not, important, unimportant, X_normed, X_x0_normed, important_rad, unimportant_rad, p, o,
                debugger_matrix, sig_radius_factor)
            if len(filtered_neighbors) > best[0]: best = (len(filtered_neighbors), unimportant_rad, important_rad)

            if len(filtered_neighbors) >= n_neighbors:
                unimportant_range, important_range = housekeeping(
                    neighbor_matrix, p, o, filtered_neighbors, verbose, input_names, y_names, unimportant_rad,
                    important_rad, unimportant_range, important_range, confound_matrix, X_x0_normed, X_normed,
                    important, unimportant, radii_matrix)
            unimportant_rad += unimp_rad_increment * scale

        important_rad += 10 ** magnitude
    return unimportant_range, important_range


def check_possible_confounding(filtered_neighbors, X_x0_normed, X_normed, input_names, p):
    """
    a param is considered a possible confound if its count is greater than that of the query param.

    sets up the second heatmap in the plot function, so it looks at three things: 1) confound 2) confound, but the
    parameter in the parameter/output pair was considered important to the output by DT, and 3) no neighbors found
    for param/output pair
    """
    # create dict with k=input, v=count of times that input var was the max perturbation in a point in the neighborhood
    max_inp_indices = {}
    for index in filtered_neighbors:
        diff = np.abs(X_x0_normed - X_normed[index])
        max_index = np.argmax(diff)
        if max_index in max_inp_indices:
            max_inp_indices[max_index] += 1
        else:
            max_inp_indices[max_index] = 1
    # print counts and keep a list of possible confounds to be checked later

    query_param_count = max_inp_indices[p] if p in max_inp_indices.keys() else 0
    possible_confound = []
    print("Count of greatest perturbation for each point in set of neighbors:")
    for k, v in max_inp_indices.items():
        print("   %s - %i" % (input_names[k], v))
        if v > query_param_count:
            possible_confound.append(k)
    return possible_confound


def update_debugger(debug_matrix, unimportant_neighbor_array, important_neighbor_array, filtered_neighbors,
                    passed_neighbors, i, o):
    unimp_set = set(unimportant_neighbor_array)
    imp_set = set(important_neighbor_array)

    debug_matrix[(i, o)]['SIG'] = filtered_neighbors
    debug_matrix[(i, o)]['ALL'] = passed_neighbors

    debug_matrix[(i, o)]['UI'] = list(unimp_set - imp_set)
    debug_matrix[(i, o)]['I'] = list(imp_set - unimp_set - set(filtered_neighbors))
    debug_matrix[(i, o)]['DIST'] = list(unimp_set & imp_set)

    return debug_matrix


def split_parameters(num_input, important_inputs, input_names, p):
    # convert str to int (idx)
    if len(important_inputs) > 0:
        input_indices = [np.where(np.array(input_names) == inp)[0][0] for inp in important_inputs]
    else:  # no important parameters
        return [x for x in range(num_input) if x != p], []

    # create subsets of the input matrix based on importance. leave out query var from the sets
    important = [x for x in input_indices if x != p]
    unimportant = [x for x in range(num_input) if x not in important and x != p]
    return unimportant, important

def check_range(input_indices, input_range, filtered_neighbors, X_x0_normed, X_normed):
    subset_X = X_normed[list(filtered_neighbors), :]
    subset_X = subset_X[:, list(input_indices)]

    max_elem = np.max(np.abs(subset_X - X_x0_normed[input_indices]))
    min_elem = np.min(np.abs(subset_X - X_x0_normed[input_indices]))

    return min(min_elem, input_range[0]), max(max_elem, input_range[1])

def print_search_output(verbose, input, output, important_rad, filtered_neighbors, unimportant_rad):
    if verbose:
        print("\nInput:", input, "/ Output:", output)
        print("Neighbors found:", len(filtered_neighbors))
        print("Max distance for important parameters: %.2f" % important_rad)
        print("Max total Euclidean distance for unimportant parameters: %.2f" % unimportant_rad)

def difficult_constraint(debug_dict, unimportant, important):
    constraint = 'UI' if debug_dict['UI'] < debug_dict['I'] + debug_dict['SIG'] else 'I'
    if (len(important)) == 0: constraint = 'UI'
    if (len(unimportant)) == 0: constraint = 'I'
    return "It was more difficult to constrain unimportant variables than important variables." if constraint == 'UI' \
        else "It was more difficult to constrain important variables than unimportant variables."

def housekeeping(neighbor_matrix, p, o, filtered_neighbors, verbose, input_names, y_names, unimportant_rad,
                 important_rad, unimportant_range, important_range, confound_matrix, X_x0_normed, X_normed,
                 important_indices, unimportant_indices, radii_matrix):
    neighbor_matrix[p][o] = filtered_neighbors
    print_search_output(verbose, input_names[p], y_names[o], important_rad, filtered_neighbors, unimportant_rad)

    unimportant_range = check_range(unimportant_indices, unimportant_range, filtered_neighbors, X_x0_normed, X_normed)
    important_range = check_range(important_indices, important_range, filtered_neighbors, X_x0_normed, X_normed)
    confound_matrix[p][o] = check_possible_confounding(filtered_neighbors, X_x0_normed, X_normed, input_names, p)
    radii_matrix[p][o] = (unimportant_rad, important_rad)
    return unimportant_range, important_range

def dd():
    # see https://stackoverflow.com/questions/16439301/cant-pickle-defaultdict
    return defaultdict(int)

#------------------redo

def redo_confounds(confound_pairs, important_inputs, y_names, max_dist, num_inputs, n_neighbors, radii_matrix,
                   input_names, X_normed, X_x0_normed, debugger_matrix, neighbor_matrix, confound_matrix, x_not,
                   unimportant_range, important_range, sig_radius_factor=2., timeout=np.inf, verbose=True):
    for i, o in confound_pairs.keys():
        confound_idxs = confound_pairs[(i, o)]
        for confound_idx in confound_idxs:
            if input_names[confound_idx] not in important_inputs[o]: important_inputs[o].append(input_names[confound_idx])
        confound_matrix[i][o] = 0.
        start = time.time()
        unimportant_range, important_range = search(
            i, o, max_dist, num_inputs, important_inputs[o], n_neighbors, radii_matrix, input_names, y_names,
            X_normed, X_x0_normed, debugger_matrix, neighbor_matrix, confound_matrix, x_not,
            unimportant_range, important_range, sig_radius_factor, timeout, verbose)
        print("--------------Took %.2f seconds" % (time.time() - start))

#------------------lsa plot

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
            if neighbor_array is not None and len(neighbor_array) > 0:
                selection = list(neighbor_array)
                X_sub = X_normed[selection, inp]  # get relevant X data points

                coef_matrix[inp][out] = abs(linregress(X_sub, y_normed[selection, out])[2])
                pval_matrix[inp][out] = linregress(X_sub, y_normed[selection, out])[3]

    return coef_matrix, pval_matrix


def create_failed_search_matrix(num_input, num_output, coef_matrix, pval_matrix, confound_matrix, input_names,
                                y_names, important_parameters, neighbor_matrix, n_neighbors, lsa_heatmap_values,
                                p_baseline=.05, confound_baseline=.5):
    """
    failure = not enough neighbors or confounded
    for each significant feature/parameter relationship identified, check if possible confounds are significant
    """
    failed_matrix = np.zeros((num_input, num_output))
    confound_dict = defaultdict(list)

    # confounded
    print("Possible confounds:")
    confound_exists = False
    for param in range(num_input):
        for feat in range(num_output):
            if pval_matrix[param][feat] < p_baseline and confound_matrix[param][feat]:
                for confound in confound_matrix[param][feat]:
                    if (coef_matrix[confound][feat] > confound_baseline and pval_matrix[confound][feat] < p_baseline)\
                            or input_names[confound] in important_parameters[feat]:
                        confound_exists = print_confound(
                            confound_exists, input_names, y_names, param, feat, confound, pval_matrix, coef_matrix)
                        failed_matrix[param][feat] = lsa_heatmap_values['confound']
                        confound_dict[(param, feat)].append(confound)
    if not confound_exists: print("None.")

    # not enough neighbors
    for param in range(num_input):
        for feat in range(num_output):
            if neighbor_matrix[param][feat] is None or len(neighbor_matrix[param][feat]) < n_neighbors:
                failed_matrix[param][feat] = lsa_heatmap_values['no_neighbors']
    return failed_matrix, confound_dict

def print_confound(confound_exists, input_names, y_names, param, feat, confound, pval_matrix, coef_matrix):
    if not confound_exists:
        print("{:30} {:30} {:30} {:20} {}".format("Independent var", "Dependent var", "Confound",
                                                  "P-val", "Abs R Coef"))
        print("----------------------------------------------------------------------------------"
              "----------------------------------------------")
        confound_exists = True
    print("{:30} {:30} {:30} {:.2e} {:20.2e}".format(
        input_names[param], y_names[feat], input_names[confound], pval_matrix[confound][feat],
        coef_matrix[confound][feat]))
    return confound_exists


def plot_sensitivity(num_input, num_output, coef_matrix, pval_matrix, input_names, y_names, sig_confounds,
                     p_baseline=.05, r_ceiling_val=None, annotated=True):
    """plot local sensitivity. mask cells with confounds and p-vals greater than than baseline
    color = sig, white = non-sig
    LGIHEST gray = no neighbors, light gray = confound but DT marked as important, dark gray = confound

    :param num_input: int
    :param num_output: int
    :param coef_matrix: 2d array of floats
    :param pval_matrix: 2d array of floats
    :param input_names: list of str
    :param y_names: list of str
    :param sig_confounds: 2d array of floats
    :return:
    """
    import seaborn as sns

    # mask confounds
    mask = np.full((num_input, num_output), True, dtype=bool)
    mask[pval_matrix < p_baseline] = False
    mask[sig_confounds != 0] = True

    # overlay relationship heatmap (hm) with confound heatmap
    fig, ax = plt.subplots(figsize=(16, 5))
    plt.title("Absolute R Coefficients", y=1.11)
    vmax = min(.7, max(.1, np.max(coef_matrix))) if r_ceiling_val is None else r_ceiling_val
    sig_hm = sns.heatmap(coef_matrix, cmap='cool', vmax=vmax, vmin=0, mask=mask, linewidths=1, ax=ax,
                         annot=annotated)
    mask[pval_matrix < p_baseline] = True
    mask[sig_confounds != 0] = False
    failed_hm = sns.heatmap(sig_confounds, cmap='Greys', vmax=1, vmin=0, mask=mask, linewidths=1, ax=ax, cbar=False)
    sig_hm.set_xticklabels(y_names)
    sig_hm.set_yticklabels(input_names)
    plt.xticks(rotation=-90)
    plt.yticks(rotation=0)
    create_LSA_custom_legend(ax)
    plt.show()

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

def create_LSA_custom_legend(ax, colormap='cool'):
    nonsig = plt.Line2D((0, 1), (0, 0), color='white', marker='s', mec='k', mew=.5, linestyle='')
    no_neighbors = plt.Line2D((0, 1), (0, 0), color='#f3f3f3', marker='s', linestyle='')
    sig_but_confounded = plt.Line2D((0, 1), (0, 0), color='#b2b2b2', marker='s', linestyle='')
    sig = LineCollection(np.zeros((2, 2, 2)), cmap=colormap, linewidth=5)
    labels = ["Not significant",  "No neighbors",  "Confounded", "Significant without confounds"]
    ax.legend([nonsig, no_neighbors, sig_but_confounded, sig], labels,
              handler_map={sig: HandlerColorLineCollection(numpoints=4)}, loc='upper center',
              bbox_to_anchor=(0.5, 1.12), ncol=5, fancybox=True, shadow=True)

#------------------plot importance via ensemble

def plot_gini(X, y, num_input, num_output, input_names, y_names, inp_out_same):
    import seaborn as sns
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

        imp_loc = list(set(np.where(rf.feature_importances_ >= baseline)[0]) | accept_outliers(rf.feature_importances_))
        feat_imp = rf.feature_importances_
        if inp_out_same:
            # imp_loc = [x + 1 if x >= i else x for x in imp_loc]
            feat_imp = np.insert(feat_imp, i, np.NaN)
        input_importances[:, i] = feat_imp
        # important_inputs[i] = list(input_names[imp_loc])

    fig, ax = plt.subplots()
    hm = sns.heatmap(input_importances, cmap='cool', fmt=".1f", linewidths=1, ax=ax, cbar=True, annot=True)
    hm.set_xticklabels(y_names)
    hm.set_yticklabels(input_names)
    plt.xticks(rotation=-90)
    plt.yticks(rotation=0)
    plt.title('Gini importances')
    plt.show()

#------------------user input prompts

def prompt_neighbor_dialog(num_input, num_output, important_inputs, input_names, y_names, X_normed,
                           x_not, verbose, n_neighbors, max_dist, inp_out_same, sig_radius_factor, timeout):
    """at the end of neighbor search, ask the user if they would like to change the starting variables"""
    while True:
        neighbor_matrix, confound_matrix, debugger_matrix, radii_matrix, unimportant_range, important_range \
            = compute_neighbor_matrix(num_input, num_output, important_inputs, input_names, y_names, X_normed,
                                      x_not, verbose, n_neighbors, max_dist, inp_out_same, sig_radius_factor, timeout)
        user_input = ''
        while user_input.lower() not in ['y', 'n', 'yes', 'no']:
            user_input = input('Was this an acceptable outcome (y/n)? ')
        if user_input.lower() in ['y', 'yes']:
            break
        elif user_input.lower() in ['n', 'no']:
            max_dist, n_neighbors, sig_radius_factor = reprompt()

    return neighbor_matrix, confound_matrix, debugger_matrix, radii_matrix, unimportant_range, important_range

def prompt_plotting(alpha, r_ceiling, annotated, confound_baseline, y_norm, global_y_norm):
    user_input = ''
    while user_input.lower() not in ['y', 'yes', 'n', 'no']:
        user_input = input('Do you want to replot the figure with new plotting parameters (alpha value, '
                           'R ceiling, confound baseline, etc)?: ')
    if user_input.lower() in ['y', 'yes']:
        y_norm, global_norm = prompt_change_y_norm(y_norm)
        return prompt_alpha(), prompt_r_ceiling_val(), prompt_annotated(), prompt_confound_baseline(), \
               y_norm, global_norm, True
    else:
        return alpha, r_ceiling, annotated, confound_baseline, y_norm, global_y_norm, False

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

def prompt_values():
    """initial prompt for variable values"""
    n_neighbors = 60
    max_dist = .01

    user_input = input('Do you want to specify the values for neighbor search? The default values are num '
                       'neighbors = 60, and starting radius for important independent variables = .01. (y/n) ')
    if user_input.lower() in ['y', 'yes']:
        n_neighbors = prompt_num_neighbors()
        max_dist = prompt_max_dist()
    elif user_input.lower() in ['n', 'no']:
        print('Thanks.')
    else:
        while user_input.lower not in ['y', 'yes', 'n', 'no']:
            user_input = input('Please enter y or n. ')

    return n_neighbors, max_dist

def prompt_num_neighbors():
    num_neighbors = ''
    while num_neighbors is not int:
        try:
            num_neighbors = input('Threshold for number of neighbors?: ')
            return int(num_neighbors)
        except ValueError:
            print('Please enter an integer.')
    return 60

def prompt_max_dist():
    max_dist = None
    while max_dist is not float or max_dist > .3: # magic num
        try:
            max_dist = input('Starting radius for important independent variables? Must be less than 0.3: ')
            return float(max_dist)
        except ValueError:
            print('Please enter a float.')
    return .01

def reprompt():
    """only reprompt the relevant variables"""
    return prompt_max_dist(), prompt_num_neighbors(), prompt_sig_radius_factor()

def prompt_indiv(valid_names):
    user_input = ''
    while user_input != 'best' and user_input not in valid_names:
        print('Valid strings for x0: ', ['best'] + valid_names)
        user_input = (input('Specify x0: ')).lower()

    return user_input

def prompt_feat_or_obj():
    user_input = ''
    while user_input.lower() not in ['f', 'o', 'features', 'objectives', 'feature', 'objective', 'feat', 'obj']:
        user_input = input('Do you want to analyze features or objectives?: ')
    return user_input.lower() in ['f', 'features', 'feature', 'feat']

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

def prompt_sig_radius_factor():
    factor = None
    while factor is not float:
        try:
            factor = input('By what factor should the perturbation of the independent query variable be greater than '
                           'that of the important dependent variables during neighbor search? Default is 2.: ')
            return float(factor)
        except ValueError:
            print('Please enter a float.')
    return 2.

def prompt_no_lsa():
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

def prompt_annotated():
    user_input = ''
    while user_input not in ['y', 'yes', 'n', 'no']:
        user_input = (input('Do you want the plot to be annotated? (y/n): ')).lower()
    return user_input in ['y', 'yes']

def check_user_importance_dict_correct(dct, input_names, y_names):
    incorrect_strings = []
    for y_name in dct.keys():
        if y_name not in y_names: incorrect_strings.append(y_names)
    for _, known_important_inputs in dct.items():
        if not isinstance(known_important_inputs, list):
            raise RuntimeError('For the known important variables dictionary, the value must be a list, even if '
                               'the list contains only one variable.')
        for name in known_important_inputs:
            if name not in input_names: incorrect_strings.append(name)
    if len(incorrect_strings) > 0:
        raise RuntimeError('Some strings in the known important variables dictionary are incorrect. Are the keys '
                           'dependent variables (string) and the values dependent variables (list of strings)? These '
                           'inputs have errors: %s.' % incorrect_strings)

def prompt_redo_confounds():
    user_input = ''
    while user_input not in ['yes', 'y', 'n', 'no']:
        user_input = (input('For the cells in the plot that are confounded, do you want to redo neighbor search '
                            'by constraining the confound variable as an important variable? (y/n): ')).lower()
    return user_input in ['yes', 'y']

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
    full_path = save_path + '{}_{}_{}_{}_{}_{}_perturbations'.format(*time.localtime())
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



class LSA(object):
    def __init__(self, pop=None, neighbor_matrix=None, coef_matrix=None, pval_matrix=None, sig_confounds=None,
                 input_id2name=None, y_id2name=None, X=None, y=None, x0_idx=None, important_inputs=None,
                 processed_data_y=None, crossing_y=None, z_y=None, pure_neg_y=None, n_neighbors=None,
                 confound_matrix=None, lsa_heatmap_values=None, file_path=None):
        if file_path is not None:
            self._load(file_path)
        else:
            self.neighbor_matrix = neighbor_matrix
            self.coef_matrix = coef_matrix
            self.pval_matrix = pval_matrix
            self.sig_confounds = sig_confounds
            self.X = X
            self.y = y
            self.x0_idx = x0_idx
            self.lsa_heatmap_values = lsa_heatmap_values
            self.important_inputs = important_inputs
            self.summed_obj = sum_objectives(pop, X.shape[0])

            self.processed_data_y = processed_data_y
            self.crossing_y = crossing_y
            self.z_y = z_y
            self.pure_neg_y = pure_neg_y
            self.n_neighbors = n_neighbors
            self.confound_matrix = confound_matrix

            self.input_names = input_id2name
            self.y_names = y_id2name
            self.input_name2id = {}
            self.y_name2id = {}

            for i, name in enumerate(input_id2name): self.input_name2id[name] = i
            for i, name in enumerate(y_id2name): self.y_name2id[name] = i


    def plot_colormap(self, dep_norm='none', global_log_dep=None, r_ceiling_val=.7, p_baseline=.05, confound_baseline=.5,
                      annotated=True):
        if self.neighbor_matrix is None:
            raise RuntimeError("LSA was not done.")
        interactive_colormap(dep_norm, global_log_dep, self.processed_data_y, self.crossing_y, self.z_y, self.pure_neg_y,
                             self.neighbor_matrix, self.X, self.y, self.confound_matrix, self.input_names, self.y_names,
                             self.important_inputs, self.n_neighbors, self.lsa_heatmap_values, p_baseline,
                             confound_baseline, r_ceiling_val, annotated)

    def plot_indep_vs_dep_filtered(self, input_name, y_name):
        input_id = get_var_idx(input_name, self.input_name2id)
        y_id = get_var_idx(y_name, self.y_name2id)

        if self.neighbor_matrix is None:
            raise RuntimeError("LSA was not run. Please use plot_vs_unfiltered() instead.")
        neighbor_indices = self.neighbor_matrix[input_id][y_id]
        if neighbor_indices is None or len(neighbor_indices) <= 1:
            print("No neighbors-- nothing to show.")
        else:
            a = self.X[neighbor_indices, input_id]
            b = self.y[neighbor_indices, y_id]
            plt.scatter(a, b)
            plt.scatter(self.X[self.x0_idx, input_id], self.y[self.x0_idx, y_id], color='red', marker='+')
            fit_fn = np.poly1d(np.polyfit(a, b, 1))
            plt.plot(a, fit_fn(a), color='red')

            if self.sig_confounds[input_id][y_id] == self.lsa_heatmap_values['confound']:
                if is_important(input_name, self.important_inputs):
                    plt.title("{} vs {} with p-val of {:.2e} and R coef of {:.2e}. Confounded but deemed globally "
                              "important.".format(input_name, y_name, self.pval_matrix[input_id][y_id],
                                                  self.coef_matrix[input_id][y_id]))
                else:
                    plt.title("{} vs {} with p-val of {:.2e} and R coef of {:.2e}. Confounded.".format(
                        input_name, y_name, self.pval_matrix[input_id][y_id], self.coef_matrix[input_id][y_id]))

            else:
                plt.title("{} vs {} with p-val of {:.2e} and R coef of {:.2e}. Not confounded.".format(
                    input_name, y_name, self.pval_matrix[input_id][y_id], self.coef_matrix[input_id][y_id]))

            plt.xlabel(input_name)
            plt.ylabel(y_name)
            plt.show()


    def plot_vs_unfiltered(self, x_axis, y_axis, num_models=None, last_third=False):
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

    def return_filtered_data(self, input_name, y_name):
        input_id = get_var_idx(input_name, self.input_name2id)
        y_id = get_var_idx(y_name, self.y_name2id)
        neighbor_indices = self.neighbor_matrix[input_id][y_id]
        if neighbor_indices is None or len(neighbor_indices) <= 1:
            raise RuntimeError("No neighbors were found for this pair.")
        return self.X[neighbor_indices], self.y[neighbor_indices]

    # just in case user doesn't know what pickling is
    def save(self, file_path='LSAobj.pkl'):
        if os.path.exists(file_path):
            raise RuntimeError("File already exists. Please delete the old file or give a new file path.")
        else:
            with open(file_path, 'wb') as output:
                pickle.dump(self, output, -1)

    def _load(self, pkl_path):
        with open(pkl_path, 'rb') as inp:
            storage = pickle.load(inp)
            self.neighbor_matrix = storage.neighbor_matrix
            self.coef_matrix = storage.coef_matrix
            self.pval_matrix = storage.pval_matrix
            self.sig_confounds = storage.sig_confounds
            self.X = storage.X
            self.y = storage.y

            self.processed_data_y = storage.processed_data_y
            self.crossing_y = storage.crossing_y
            self.z_y = storage.z_y
            self.pure_neg_y = storage.pure_neg_y
            self.n_neighbors = storage.n_neighbors
            self.confound_matrix = storage.confound_matrix

            self.input_names = storage.input_names
            self.y_names = storage.y_names
            self.lsa_heatmap_values = storage.lsa_heatmap_values
            self.important_inputs = storage.important_inputs
            self.summed_obj = storage.summed_obj
            self.input_name2id = storage.input_name2id
            self.y_name2id = storage.y_name2id


class InterferencePlot(object):
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
    def __init__(self, debug_matrix=None, X=None, y=None, input_id2name=None, y_id2name=None, important_inputs=None,
                 radii_matrix=None, file_path=None):
        """

        :param debug_matrix: actually a dict (key=input id) of dicts (key=output id) of lists of tuples of the form
        (array representing point in input space, string representing category)
        :param y_id2name:
        """
        if file_path is not None:
            self._load(file_path)
        else:
            self.debug_matrix = debug_matrix
            self.radii_matrix = radii_matrix
            self.X = X
            self.y = y
            self.input_id2name = input_id2name
            self.important_inputs = important_inputs
            self.input_name2id = {}
            self.y_name2id = {}
            default_alpha = .3
            self.cat2color = {'UI': 'red', 'I': 'xkcd:muddy yellow', 'DIST': 'purple', 'SIG': 'lawngreen',
                              'ALL': 'xkcd:dark blue grey'}
            self.cat2alpha = {'UI' : default_alpha, 'I' : default_alpha, 'DIST': default_alpha, 'SIG' : default_alpha,
                              'ALL' : default_alpha}
            self.previous_plot_data = defaultdict(dict)

            for i, name in enumerate(input_id2name): self.input_name2id[name] = i
            for i, name in enumerate(y_id2name): self.y_name2id[name] = i


    def plot_PCA(self, input_name, y_name, alpha_vals=None):
        """try visualizing all of the input variable values by flattening it"""
        all_points_X, _, cat2idx = extract_data(input_name, y_name, self.previous_plot_data, self.X, self.y, self.debug_matrix,
                                           self.input_name2id, self.y_name2id)
        if all_points_X is not None:
            if alpha_vals is not None: self.cat2alpha = modify_alpha_vals(alpha_vals, self.cat2alpha)
            pca = PCA(n_components=2)
            pca.fit(all_points_X)
            flattened = pca.transform(all_points_X)

            for cat in self.cat2color:
                idxs = cat2idx[cat]
                plt.scatter(flattened[idxs, 0], flattened[idxs, 1], c=self.cat2color[cat], label=cat,
                            alpha=self.cat2alpha[cat])
            plt.legend(labels=list(self.cat2color.keys()))
            plt.xlabel('Principal component 1 (%.3f)' % pca.explained_variance_ratio_[0])
            plt.ylabel('Principal component 2 (%.3f)' % pca.explained_variance_ratio_[1])
            plt.title('Neighbor search for the sensitivity of %s to %s' % (y_name, input_name))
            plt.show()
        else:
            print("No neighbors-- nothing to show.")


    def plot_vs(self, input_name, y_name, z1, z2, alpha_vals=None):
        """plot one input variable vs another input"""
        z1_idx, input_bool_z1 = get_var_idx_agnostic(z1, self.input_name2id, self.y_name2id)
        z2_idx, input_bool_z2 = get_var_idx_agnostic(z2, self.input_name2id, self.y_name2id)
        all_points_X, all_points_y, cat2idx = extract_data(input_name, y_name, self.previous_plot_data, self.X, self.y,
                                                           self.debug_matrix, self.input_name2id, self.y_name2id)
        if alpha_vals is not None: self.cat2alpha = modify_alpha_vals(alpha_vals, self.cat2alpha)
        for cat in self.cat2color:
            idxs = cat2idx[cat]
            a = get_column(z1_idx, input_bool_z1, all_points_X, all_points_y)
            b = get_column(z2_idx, input_bool_z2, all_points_X, all_points_y)
            plt.scatter(a[idxs], b[idxs], c=self.cat2color[cat], label=cat, alpha=self.cat2alpha[cat])
        plt.legend(labels=list(self.cat2color.keys()))
        plt.xlabel(z1)
        plt.ylabel(z2)
        plt.title('Neighbor search for the sensitivity of %s to %s' % (y_name, input_name))
        plt.show()


    def get_interference_by_classification(self, input_name, y_name, class_0=None, class_1=None):
        all_points_X,_, cat2idx = extract_data(input_name, y_name, self.previous_plot_data, self.X, self.y,
                                               self.debug_matrix, self.input_name2id, self.y_name2id)
        if class_0 is None and class_1 is None:
            class_0 = [x for x in cat2idx.keys() if x != 'ALL']
            class_1 = ['ALL']
        else:
            check_classes(class_0, class_1, cat2idx)
        if all_points_X is None:
            print('No neighbors found.')
        else:
            y_labels = np.zeros(all_points_X.shape[0])
            for cat in class_1:
                for idx in cat2idx[cat]: y_labels[idx] = 1 #cat2idx[cat]
            all_idx = set()
            for binary_class in [class_0, class_1]:
                for cat in binary_class:
                    all_idx = all_idx | set(cat2idx[cat])
            y_labels = y_labels[list(all_idx)]
            X = all_points_X[list(all_idx)]

            if np.all(y_labels == 0) or np.all(y_labels == 1):
                print('Could not calculate interference; one of the classes has 0 data points.')
            else:
                dt = ExtraTreesRegressor(random_state=0, max_features=max(1, int(.1 * X.shape[1])), max_depth=25,
                                         n_estimators=50)
                dt.fit(X, y_labels)

                input_list = list(zip([round(t, 4) for t in dt.feature_importances_], list(self.input_name2id.keys())))
                input_list.sort(key=lambda x: x[0], reverse=True)
                if len(input_list) < 5:
                    print('The top informative input variables that indicated failure (based on Gini importance) '
                          'were: ', input_list[:5])
                else:
                    print('The top five most informative input variables that indicated failure (based on Gini '
                          'importance) were: ', input_list[:5])


    def get_interference_manually(self, input_name, y_name):
        threshold_factor = 2.5
        all_points_X, _, cat2idx = extract_data(input_name, y_name, self.previous_plot_data, self.X, self.y,
                                                self.debug_matrix, self.input_name2id, self.y_name2id)
        if all_points_X is None:
            print('No neighbors found.')
        else:
            print_search_stats(all_points_X, cat2idx, input_name)

            x_idx = get_var_idx(input_name, self.input_name2id)
            y_idx = get_var_idx(y_name, self.y_name2id)
            imp_idx = [self.input_name2id[key] for key in self.important_inputs[y_idx]]
            unimp_idx = [x for x in range(all_points_X.shape[1]) if x not in imp_idx \
                         and x in range(len(self.input_name2id))] #exclude imp ind var and dependent var
            if x_idx in imp_idx: imp_idx.remove(x_idx)
            if x_idx in unimp_idx: unimp_idx.remove(x_idx)

            count_arr = np.zeros((len(self.input_name2id), 1))
            count_arr = count_imp_inteference(count_arr, cat2idx, all_points_X, self.radii_matrix, x_idx, y_idx, imp_idx)
            count_arr = count_unimp_inference(count_arr, cat2idx, all_points_X, self.radii_matrix, x_idx, y_idx, unimp_idx,
                                              threshold_factor)
            count_arr[x_idx] = len(cat2idx['DIST'])

            print_interference_ratios(count_arr, x_idx, self.input_id2name)

    def save(self, file_path='LSAobj.pkl'):
        if os.path.exists(file_path):
            raise RuntimeError("File already exists. Please delete the old file or give a new file path.")
        else:
            with open(file_path, 'wb') as output:
                pickle.dump(self, output, -1)

    def _load(self, pkl_path):
        with open(pkl_path, 'rb') as inp:
            storage = pickle.load(inp)
            self.debug_matrix = storage.debug_matrix
            self.radii_matrix = storage.radii_matrix
            self.X = storage.X
            self.y = storage.y
            self.input_id2name = storage.input_id2name
            self.important_inputs = storage.important_inputs
            self.input_name2id = storage.input_name2id
            self.y_name2id = storage.y_name2id
            self.cat2color = storage.cat2color
            self.cat2alpha = storage.cat2alpha
            self.previous_plot_data = storage.previous_plot_data


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

def is_important(input_name, important_inputs):
    return len(np.where(important_inputs == input_name)[0]) > 0

def get_points(input_name, y_name, debug_matrix, input_name2id, y_name2id):
    try:
        buckets = debug_matrix[(input_name2id[input_name], y_name2id[y_name])]
    except:
        raise RuntimeError('At least one provided variable name is incorrect. For input variables, valid choices are: '
                           '%s. For output variables: %s.' % (list(input_name2id.keys()), list(y_name2id.keys())))
    return buckets

def extract_data(input_name, y_name, previous_plot_data, X, y, debug_matrix, input_name2id, y_name2id):
    if input_name in previous_plot_data and y_name in previous_plot_data[input_name]:
        all_points_X = previous_plot_data[input_name][y_name][0]
        all_points_y = previous_plot_data[input_name][y_name][1]
        cat2idx = previous_plot_data[input_name][y_name][2]
    else:
        buckets = get_points(input_name, y_name, debug_matrix, input_name2id, y_name2id)
        all_points_X = None
        all_points_y = None
        cat2idx = defaultdict(list)
        idx_counter = 0
        for cat, idx in buckets.items():
            #idx_list = list(idx) #for some categories, idx is a set
            if len(idx) == 0: continue
            all_points_X = X[idx] if all_points_X is None else np.concatenate((all_points_X, X[idx]))
            all_points_y = y[idx] if all_points_y is None else np.concatenate((all_points_y, y[idx]))
            cat2idx[cat] = list(range(idx_counter, idx_counter + len(idx)))
            idx_counter += len(idx)
        previous_plot_data[input_name][y_name] = (all_points_X, all_points_y, cat2idx)
    return all_points_X, all_points_y, cat2idx

def get_column(idx, X_bool, X, y):
    if X_bool:
        return X[:, idx]
    else:
        return y[:, idx]

def sum_objectives(pop, n):
    summed_obj = np.zeros((n,))
    counter = 0
    generation_array = pop.history
    for generation in generation_array:
        for datum in generation:
            summed_obj[counter] = sum(abs(datum.objectives))
            counter += 1
    return summed_obj

def count_imp_inteference(count_arr, cat2idx, all_points, radii_matrix, x_idx, y_idx, imp_idx):
    for cat in ['SIG', 'I']:
        for point_idx in cat2idx[cat]:
            failed_idx = [x for x in np.where(all_points[point_idx] > radii_matrix[x_idx][y_idx][1])[0] \
                          if x in imp_idx]
            count_arr[failed_idx] += 1
    return count_arr

def count_unimp_inference(count_arr, cat2idx, all_points, radii_matrix, x_idx, y_idx, unimp_idx, threshold_factor):
    threshold = (radii_matrix[x_idx][y_idx][0] / len(unimp_idx)) * threshold_factor
    # pass unimp but not imp
    for point_idx in cat2idx['UI']:
        failed_idx = [x for x in np.where(all_points[point_idx] > threshold)[0] if x in unimp_idx]
        count_arr[failed_idx] += 1

    return count_arr

def print_interference_ratios(count_arr, x_idx, input_id2name):
    ratios = count_arr / np.sum(count_arr)
    rank_idx = rankdata(-ratios, method='ordinal') - 1  # descending order
    sorted_ratios = sorted(ratios, reverse=True)
    for i in range(len(sorted_ratios)):
        j = np.where(rank_idx == i)[0][0]
        print('%s: %.3f' % (input_id2name[j], sorted_ratios[i]))

def print_search_stats(all_points, cat2idx, input_name):
    print('Out of %d points that passed the important distance filter, %d had significant perturbations in the '
          'direction of %s.' % (all_points.shape[0] - len(cat2idx['UI']), (len(cat2idx['SIG']) + len(cat2idx['ALL'])),
                                input_name))
    print("%d points passed only the important radius filter, %d passed only the unimportant radius filter, and "
          "%d passed all criteria." % (len(cat2idx['SIG']) + len(cat2idx['I']), len(cat2idx['UI']),
                                       len(cat2idx['ALL'])))

def modify_alpha_vals(alpha_vals, cat2alpha):
    for cat in alpha_vals.keys():
        if cat in cat2alpha.keys():
            cat2alpha[cat] = alpha_vals[cat]
    return cat2alpha

def check_classes(class_0, class_1, cat2idx):
    if class_0 is None or class_1 is None:
        raise RuntimeError("Please specify both classes instead of just one.")
    if not isinstance(class_0, list) or not isinstance(class_1, list):
        raise RuntimeError("Classes must be specified as a list.")
    for binary_class in [class_0, class_1]:
        for cat in binary_class:
            if cat not in cat2idx.keys():
                raise RuntimeError("%s is an incorrect category. Possible choices are: %s." % (cat, list(cat2idx.keys())))
