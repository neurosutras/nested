from utils import *
from nested import *


def complex_problem(parameters):
    """

    :param parameters: array
    :return: dict
    """
    features = {}
    num_params = len(parameters)
    f1 = parameters[0]
    features['f1'] = f1
    g = 1. + 9. / (num_params - 1.) * np.sum(parameters[1:])
    features['g'] = g
    h = 1. - np.sqrt(f1 / g)
    features['h'] = h
    return features


def get_objectives(features):
    """

    :param features: dict
    :return: dict
    """
    objectives = {}
    for feature_name in ['f1', 'g', 'h']:
        objective_name = feature_name
        objectives[objective_name] = features[feature_name]
    f2 = features['g'] * features['h']
    objectives['f2'] = f2
    return objectives


num_params = 30
param_names = ['x%i' % i for i in range(num_params)]
bounds = [(0., 1.) for i in range(num_params)]
x0 = [0.5 * (xmin + xmax) for (xmin, xmax) in bounds]
feature_names = ['g', 'h']
objective_names = ['f1', 'f2', 'g', 'h']
get_features = complex_problem
get_objectives = get_objectives

path_length = 2
wrap_bounds = False
max_iter = 30
# max_iter = 40
hot_start = None
# hot_start = 'data/071220172044_bgen_example_storage.hdf5'


storage_file_path = 'data/%s_pop_anneal_example_storage.hdf5' % (datetime.datetime.today().strftime('%m%d%Y%H%M'))

pop_anneal = PopulationAnnealing(param_names, feature_names, objective_names, 100, x0=x0, bounds=bounds,
                           wrap_bounds=wrap_bounds, seed=0,
            max_iter=max_iter, path_length=path_length, adaptive_step_factor=0.9, survival_rate=0.20, disp=True,
            hot_start=hot_start, select='select_survivors_by_rank_and_fitness')

offset = pop_anneal.num_gen
for i, param_list in enumerate(pop_anneal()):
    if i + offset > 0 and (i + offset) % path_length == 0:
        pop_anneal.storage.save(storage_file_path, n=path_length)
    features = map(get_features, param_list)
    objectives = map(get_objectives, features)
    pop_anneal.update_population(features, objectives)
pop_anneal.storage.save(storage_file_path, n=path_length)
pop_anneal.storage.plot()


