from nested.optimize_utils import *
import click


context = Context()


def complex_problem(parameters, export=False):
    """

    :param parameters: array
    :return: dict
    """
    print('Process: %i evaluating parameters: %s' % (os.getpid(), ', '.join('%.3f' % x for x in parameters)))
    sys.stdout.flush()

    # Test handling of failure to compute required feature
    if parameters[0] > 1.:
        return dict()

    features = {}
    num_params = len(parameters)
    f1 = parameters[0]
    features['f1'] = f1
    g = 1. + 9. / (num_params - 1.) * np.sum(parameters[1:])
    features['g'] = g
    h = 1. - np.sqrt(f1 / g)
    features['h'] = h

    return features


def get_objectives(features, export=False):
    """

    :param features: dict
    :param export: bool
    :return: tuple of dict
    """
    objectives = {}
    for feature_name in ['f1', 'g', 'h']:
        if feature_name not in features:
            return dict(), dict()
        objective_name = feature_name
        objectives[objective_name] = features[feature_name]
    f2 = features['g'] * features['h']
    objectives['f2'] = f2
    return features, objectives


def test_shared_features(parameters, export=False):
    return {'shared_features': 1.}


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True,))
@click.option("--num-params", type=int, default=10)
@click.option("--hot-start", is_flag=True)
@click.option("--storage-file-path", type=str, default=None)
@click.option("--plot", is_flag=True)
@click.pass_context
def main(cli, num_params, hot_start, storage_file_path, plot):
    """
    :param cli: :class:'click.Context': used to process/pass through unknown click arguments
    :param num_params:
    :param hot_start: bool
    :param storage_file_path: str (path)
    :param plot: bool
    """
    context.update(locals())
    kwargs = get_unknown_click_arg_dict(cli.args)
    param_names = ['x%i' % i for i in range(num_params)]
    bounds = [(0., 1.) for i in range(num_params)]
    bounds[0] = (0., 1.1)
    x0 = [0.5 * (xmin + xmax) for (xmin, xmax) in bounds]
    feature_names = ['g', 'h']
    objective_names = ['f1', 'f2', 'g', 'h']
    get_features = complex_problem

    if storage_file_path is None:
        storage_file_path = 'data/%s_pop_anneal_example_storage.hdf5' % \
                            (datetime.datetime.today().strftime('%Y%m%d_%H%M'))

    pop_anneal = PopulationAnnealing(param_names=param_names, feature_names=feature_names,
                                     objective_names=objective_names, x0=x0, bounds=bounds, seed=0, disp=True,
                                     hot_start=hot_start, storage_file_path=storage_file_path, **kwargs)

    features = [{} for pop_id in range(pop_anneal.pop_size)]
    objectives = [{} for pop_id in range(pop_anneal.pop_size)]
    context.update(locals())

    for generation in pop_anneal():
        new_features = list(map(get_features, generation))
        for pop_id, this_features in enumerate(new_features):
            features[pop_id].update(this_features)
        primitives = list(map(get_objectives, features))
        for pop_id, (this_features, this_objectives) in enumerate(primitives):
            features[pop_id].update(this_features)
            objectives[pop_id].update(this_objectives)
        pop_anneal.update_population(features, objectives)

    if plot:
        pop_anneal.storage.plot()
    context.update(locals())


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
