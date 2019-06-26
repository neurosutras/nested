from nested.utils import *
from nested.optimize_utils import PopulationStorage, Individual, HallOfFame
import collections
import numpy as np
from collections import defaultdict
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import BallTree
from sklearn.decomposition import PCA
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.collections import LineCollection
import math
import warnings


def local_sensitivity(population, x0_string=None, input_str=None, output_str=None, no_LSA=None,relaxed_bool=None,
                      relaxed_factor=1., norm_search=None, n_neighbors=None, max_dist=None, verbose=True, save_path=''):
    """main function for plotting and computing local sensitivity
    note on variable names: X_x0 redundantly refers to the parameter values associated with the point x0. x0 by itself
    refers to both the parameters and the output
    input = independent var, output = dependent var

    :param population: PopulationStorage object
    :param verbose: bool. if True, will print radius and num neighbors for each parameter/objective pair
    :param save_path: str for where perturbation vector will be saved if generated
    :return:
    """
    #acceptable strings
    FEAT_STRINGS = ['f', 'feature', 'features']
    OBJ_STRINGS = ['o', 'objective', 'objectives']
    PARAM_STRINGS = ['parameter', 'p', 'parameters']

    #prompt user
    if x0_string is None: x0_string = prompt_indiv(population)
    if input_str is None: input_str = prompt_input()
    if output_str is None: output_str = prompt_output()
    if no_LSA is None: no_LSA = prompt_no_LSA()
    if relaxed_bool is None: relaxed_bool = prompt_DT_constraint() if not no_LSA else False
    if relaxed_bool and relaxed_factor == 1: relaxed_factor = prompt_relax_constraint()
    if norm_search is None: norm_search = prompt_linspace()
    feat_bool = output_str in FEAT_STRINGS
    if n_neighbors is None and max_dist is None: n_neighbors, max_dist = prompt_values()
    if max_dist is None: max_dist = prompt_max_dist()
    if n_neighbors is None: n_neighbors = prompt_num_neighbors()

    #set variables based on user input
    input_names, y_names = get_variable_names(population, input_str, output_str, OBJ_STRINGS, FEAT_STRINGS,
                                              PARAM_STRINGS)
    num_param = len(population.param_names)
    num_input = len(input_names)
    num_output = len(y_names)
    input_is_not_param = input_str not in PARAM_STRINGS
    inp_out_same = (input_str in FEAT_STRINGS and output_str in FEAT_STRINGS) or \
                   (input_str in OBJ_STRINGS and output_str in OBJ_STRINGS)

    #process and potentially normalize data
    data = pop_to_matrix(population, feat_bool)
    processed_data, crossing, z = process_data(data)
    data_normed, x0_normed, packaged_variables = normalize_data(population, data, processed_data, crossing, z, x0_string,
            population.param_names, input_is_not_param, norm_search)

    important_inputs, dominant_list = get_important_inputs(data_normed, num_input, num_output, num_param, input_names, y_names,
                                            input_is_not_param, inp_out_same, relaxed_factor)

    if no_LSA:
        lsa_obj = LSA(None, None, None, None, input_names, y_names, data_normed)
        print("No exploration vector generated.")
        return None, lsa_obj, None

    X_x0 = packaged_variables[0]; scaling = packaged_variables[1]; logdiff_array = packaged_variables[2]
    logmin_array = packaged_variables[3]; diff_array = packaged_variables[4]; min_array = packaged_variables[5]
    X_normed = data_normed[:, :num_param] if input_str in PARAM_STRINGS else data_normed[:, num_param:]
    y_normed = data_normed[:, num_param:]

    #LSA
    neighbor_matrix, confound_matrix, debugger_matrix = prompt_neighbor_dialog(num_input, num_output, num_param,
        important_inputs, input_names, y_names, X_normed, x0_normed, verbose, n_neighbors, max_dist, input_is_not_param,
        inp_out_same, dominant_list)

    coef_matrix, pval_matrix = get_coef(num_input, num_output, neighbor_matrix, X_normed, y_normed)
    sig_confounds = determine_confounds(num_input, num_output, coef_matrix, pval_matrix, confound_matrix,
                                        input_names, y_names, important_inputs, neighbor_matrix)

    #create objects to return
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


#------------------processing populationstorage and normalizing data

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


def x0_to_array(population, x0_string, param_names, data, processed_data):
    """
    from x0 string (e.g. 'best'), returns the respective array/data which contains
    both the parameter and output values
    """
    fame = HallOfFame(population)
    num_param = len(fame.param_names)

    x0_x_dict = fame.x_dict.get(x0_string)
    x0_x_array = order_dict(x0_x_dict, param_names.tolist())

    index = np.where(data[:, :num_param] == x0_x_array)[0][0]
    return processed_data[index, :], num_param


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

    scaling = None  # holds a list of whether the column was log or lin normalized (string)
    if norm_search:
        scaling = np.array(['log'] * num_cols)
        scaling[np.where(logdiff_array < 2)[0]] = 'lin'
        scaling[crossing] = 'lin'; scaling[z] = 'lin'
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


def order_dict(x0_dict, names):
    """
    HallOfFame = dict with dicts, therefore ordering is different from .yaml file.
    x0_dict = dict from HallOfFame: key = string (name), val = real number
    name = list of input variable names from .yaml file
    this orders the values in the way that the .yaml file is
    """
    ordered_list = [None] * len(names)
    for var_name, val in x0_dict.items():
        index = names.index(var_name)
        ordered_list[index] = val
    return np.asarray(ordered_list)

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

def get_important_inputs(data, num_input, num_output, num_param, input_names, y_names, input_is_not_param,
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


def check_dominant(feat_imp, imp_loc, unimp_loc):
    imp_mean = np.mean(feat_imp[imp_loc])
    unimp_mean = np.mean(feat_imp[unimp_loc])
    if imp_mean != 0 and unimp_mean != 0 and len(imp_loc) != 0 and int(math.log10(imp_mean)) - \
            int(math.log10(unimp_mean)) >= 2:
        return True
    return False

#------------------neighbor search

def get_potential_neighbors(important, unimportant, X_normed, X_x0_normed, important_rad, unimportant_rad):
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


def filter_neighbors(x_not, important, unimportant, X_normed, X_x0_normed, important_rad, unimportant_rad, i, o,
                     debug_matrix):
    """filter according to the radii constraints and if query parameter perturbation > twice the max perturbation
    of important parameters
    passed neighbors = passes all constraints
    filtered neighbors = neighbors that fit the important input variable distance constraint + the distance of
        the input variable of interest is more than twice that of the important variable constraint"""

    unimportant_neighbor_array, important_neighbor_array = get_potential_neighbors(important, unimportant, X_normed,
        X_x0_normed, important_rad, unimportant_rad)
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

                filtered_neighbors, debugger_matrix = filter_neighbors(x_not, important, unimportant, X_normed,
                    X_x0_normed, important_rad, unimportant_rad, p, o, debugger_matrix)

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
                    filtered_neighbors, debugger_matrix = filter_neighbors(x_not, important, unimportant, X_normed,
                    X_x0_normed, important_rad, unimportant_rad, p, o, debugger_matrix)

                    if len(filtered_neighbors) >= n_neighbors:
                        neighbor_matrix, important_range, unimportant_range, confound_matrix = housekeeping(
                            neighbor_matrix, p, o, filtered_neighbors, verbose, input_names, y_names, important_rad,
                            unimportant_rad, important_range, unimportant_range, confound_matrix, X_x0_normed,
                            X_normed, important, unimportant)
                    unimportant_rad += UNIMP_RAD_INCREMENT

                important_rad += 10 ** magnitude

    print("Important independent variable radius range:", important_range, "/ Unimportant:", unimportant_range)
    return neighbor_matrix, confound_matrix, debugger_matrix


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
    for k, v in max_inp_indices.items():
        print("   %s - %i" % (input_names[k], v))
        if v > query_param_count:
            possible_confound.append(k)
    return possible_confound


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
        print("Max total euclidean distance for unimportant parameters: %.2f" % unimportant_rad)

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
    confound_matrix[p][o] = check_possible_confounding(filtered_neighbors, X_x0_normed, X_normed, input_names, p)

    return neighbor_matrix, important_range, unimportant_range, confound_matrix


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
    CONFOUND_BASELINE = .03 #abs R value must be greater than baseline to be considered a confound

    # confound
    print("Possible confounds:")
    confound_exists = False
    for param in range(num_input):
        for feat in range(num_output):
            if pval_matrix[param][feat] < P_BASELINE and confound_matrix[param][feat]:
                for confound in confound_matrix[param][feat]:
                    if coef_matrix[confound][feat] > CONFOUND_BASELINE and pval_matrix[confound][feat] < P_BASELINE:
                        if not confound_exists:
                            print("{:30} {:30} {:30} {:20} {}".format("Independent var", "Dependent var", "Confound",
                                  "P-val", "Abs R Coef"))
                            print("----------------------------------------------------------------------------------"
                                  "----------------------------------------------")
                            confound_exists = True
                        print("{:30} {:30} {:30} {:.2e} {:20.2e}".format(input_names[param], y_names[feat], input_names[confound],
                                pval_matrix[confound][feat], coef_matrix[confound][feat]))
                        sig_confounds[param][feat] = 1
    if not confound_exists: print("None.")

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
    """not in use.
    normalize absolute coefficients by column. only normalize the ones less than the pval

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

    # mask confounds
    mask = np.full((num_input, num_output), True, dtype=bool)
    mask[pval_matrix < P_BASELINE] = False
    mask[sig_confounds != 0] = True

    # overlay relationship heatmap (hm) with confound heatmap
    fig, ax = plt.subplots(figsize=(16, 5))
    plt.title("Absolute R Coefficients", y=1.11)
    sig_hm = sns.heatmap(coef_matrix, fmt="g", cmap='cool', vmax=.3, vmin=0, mask=mask, linewidths=1, ax=ax)
    confound_hm = sns.heatmap(sig_confounds, fmt="g", cmap='Greys', vmax=1, linewidths=1, ax=ax, alpha=.3, cbar=False)
    sig_hm.set_xticklabels(y_names)
    sig_hm.set_yticklabels(input_names)
    plt.xticks(rotation=-90)
    plt.yticks(rotation=0)
    create_LSA_custom_legend(ax)
    plt.savefig('data/test.png')
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
    nonsig = plt.Line2D((0, 1), (0, 0), color='white', marker='s', mec='k', mew=1., linestyle='')
    no_neighbors = plt.Line2D((0, 1), (0, 0), color='#f6f6f6', marker='s', linestyle='')
    confound_but_imp = plt.Line2D((0, 1), (0, 0), color='#d6d6d6', marker='s', linestyle='')
    sig_but_confounded = plt.Line2D((0, 1), (0, 0), color='#b2b2b2', marker='s', linestyle='')
    sig = LineCollection(np.zeros((2, 2, 2)), cmap=colormap, linewidth=5)
    labels = ["Not significant", "No neighbors", "Confounded but deemed important", "Confounded",
              "Significant without confounds"]
    ax.legend([nonsig, no_neighbors, confound_but_imp, sig_but_confounded, sig], labels,
              handler_map={sig: HandlerColorLineCollection(numpoints=4)}, loc='upper center', bbox_to_anchor=(0.5, 1.12),
              ncol=5, fancybox=True, shadow=True)


#------------------user input prompts

def prompt_neighbor_dialog(num_input, num_output, num_param, important_inputs, input_names, y_names, X_normed,
                           x0_normed, verbose, n_neighbors, max_dist, input_is_not_param, inp_out_same, dominant_list):
    """at the end of neighbor search, ask the user if they would like to change the starting variables"""
    while True:
        neighbor_matrix, confound_matrix, debugger_matrix = compute_neighbor_matrix(num_input, num_output, num_param,
             important_inputs, input_names, y_names, X_normed, x0_normed, verbose, n_neighbors, max_dist,
             input_is_not_param, inp_out_same, dominant_list)
        user_input = ''
        while user_input.lower() not in ['y', 'n', 'yes', 'no']:
            user_input = input('Was this an acceptable outcome (y/n)? ')
        if user_input.lower() in ['y', 'yes']:
            break
        elif user_input.lower() in ['n', 'no']:
            #n_neighbors, max_dist = prompt_values()
            max_dist, n_neighbors, dominant_list = reprompt(num_input, important_inputs, input_names, dominant_list)

    return neighbor_matrix, confound_matrix, debugger_matrix

def prompt_values():
    """initial prompt for variable values"""
    n_neighbors = 60
    max_dist = .01

    user_input = input('Do you want to specify the values for neighbor search? The default values are num '
                            'neighbors = 60, and starting radius for important independent variables = .01. (y/n) ')
    if user_input in ['y', 'Y']:
        n_neighbors = prompt_num_neighbors()
        max_dist = prompt_max_dist()
    elif user_input in ['n', 'N']:
        print('Thanks.')
    else:
        while user_input not in ['y', 'Y', 'n', 'N']:
            user_input = input('Please enter y or n. ')

    return n_neighbors, max_dist

def prompt_num_neighbors():
    num_neighbors = ''
    while num_neighbors is not int:
        try:
            num_neighbors = input('Threshold for number of neighbors?: ')
            return int(num_neighbors)
        except ValueError:
            print('Please enter a number.')
    return 60

def prompt_max_dist():
    max_dist = ''
    while max_dist is not float:
        try:
            max_dist = input('Starting radius for important independent variables?: ')
            return float(max_dist)
        except ValueError:
            print('Please enter a number.')
    return .01

def reprompt(num_input, important_inputs, input_names, dominant_list):
    """only reprompt the relevant variables"""
    max_dist = prompt_max_dist()
    n_neighbors = prompt_num_neighbors()
    relaxed_bool = prompt_DT_constraint()
    if relaxed_bool:
        relaxed_factor = prompt_relax_constraint()
        if relaxed_factor:
            imp_idxs, _ = split_parameters(num_input, important_inputs, input_names, -1)
            dominant_list[imp_idxs] = relaxed_bool
    return max_dist, n_neighbors, dominant_list

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
            user_input = float(input('By what factor should the constraint be relaxed? It is currently set to 1, '
                                     'i.e., no relaxation: '))
            return float(user_input)
        except ValueError:
            print('Please enter a number.')
    return 1.


#------------------explore vector

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
            if neighbor_matrix[inp][output] is None or len(neighbor_matrix[inp][output]) < n_neighbors:
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
        DEFAULT_ALPHA = .3
        self.cat2color = {'UI': 'red', 'I': 'fuchsia', 'DIST': 'purple', 'SIG': 'lawngreen', 'ALL': 'cyan'}
        self.cat2alpha = {'UI' : DEFAULT_ALPHA, 'I' : DEFAULT_ALPHA, 'DIST': DEFAULT_ALPHA, 'SIG' : DEFAULT_ALPHA,
                          'ALL' : DEFAULT_ALPHA}
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

    def modify_alpha_vals(self, alpha_vals):
        for cat in alpha_vals.keys():
            if cat in  self.cat2alpha.keys():
                self.cat2alpha[cat] = alpha_vals[cat]

    def plot_PCA(self, input_name, y_name, alpha_vals=None):
        """try visualizing all of the input variable values by flattening it"""
        all_points, cat2idx = self.extract_data(input_name, y_name)
        if all_points is not None:
            if alpha_vals is not None: self.modify_alpha_vals(alpha_vals)
            pca = PCA(n_components=2)
            pca.fit(all_points)
            flattened = pca.transform(all_points)

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

    def plot_vs(self, input_name, y_name, x1, x2, alpha_vals=None):
        """plot one input variable vs another input"""
        try:
            x1_idx = self.input_name2id[x1]
            x2_idx = self.input_name2id[x2]
        except:
            raise RuntimeError(
                'At least one provided variable name is incorrect. For input variables, valid choices are ',
                list(self.input_name2id.keys()), '.')
        if alpha_vals is not None: self.modify_alpha_vals(alpha_vals)
        all_points, cat2idx = self.extract_data(input_name, y_name)
        for cat in self.cat2color:
            idxs = cat2idx[cat]
            plt.scatter(all_points[idxs, x1_idx], all_points[idxs, x2_idx], c=self.cat2color[cat], label=cat,
                        alpha=self.cat2alpha[cat])
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
            print('Out of %d points that passed the important distance filter, %d had significant perturbations in the '
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
