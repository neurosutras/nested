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


num_params = 10
param_names = ['x%i' % i for i in range(num_params)]
bounds = [(0., 1.) for i in range(num_params)]
x0 = [0.5 * (xmin + xmax) for (xmin, xmax) in bounds]
feature_names = ['g', 'h']
objective_names = ['f1', 'f2', 'g', 'h']
get_features = complex_problem
get_objectives = get_objectives

pop_size = 200
path_length = 3
wrap_bounds = False
max_iter = 10
hot_start = False

storage_file_path = 'data/%s_pop_anneal_example_storage.hdf5' % (datetime.datetime.today().strftime('%Y%m%d_%H%M'))

pop_anneal = PopulationAnnealing(param_names=param_names, feature_names=feature_names, objective_names=objective_names,
                                 pop_size=pop_size, x0=x0, bounds=bounds, wrap_bounds=wrap_bounds, seed=0,
                                 max_iter=max_iter, path_length=path_length, adaptive_step_factor=0.9,
                                 survival_rate=0.20, disp=True, hot_start=hot_start,
                                 storage_file_path=storage_file_path,
                                 select='select_survivors_by_rank_and_fitness')

for generation in pop_anneal():
    features = map(get_features, generation)
    objectives = map(get_objectives, features)
    pop_anneal.update_population(features, objectives)

pop_anneal.storage.plot()
