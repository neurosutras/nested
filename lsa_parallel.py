from mpi4py import MPI
import io
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import linregress
import numpy as np
from nested.lsa import *
from nested.optimize_utils import PopulationStorage

"""
to run in a jupyter notebook:
1) change jupyter to True (line 33)
2) specify x0_str, input_str, output_str, indep_norm, and dep_norm in sa.run_analysis (line 389)
    *note: if indep_norm is 'loglin', then global_log_indep must be set. same with dep_norm
     and global_log_dep
    *example: sa.run_analysis(x0_str='best', input_str='p', output_str='f', indep_norm='lin',
                              dep_norm='none')
3) then in jupyter, run
   !mpiexec -n 4 python lsa_parallel.py
4) after it's complete, the cell should have a statement like 
    "Plot object saved to data/202012715201_plot_object.pkl."
then run a cell with the following content: 
    from nested.lsa import *
    plot = load("data/202012715201_plot_object.pkl")   # your path here
    plot.plot_interactive()
"""

# specify variables here
storage_file_path = "data/20190930_1534_pa_opt_hist_test.hdf5"
save_imgs = False # for various scatter plots
save_format = 'png'
save_txt = True
verbose = True
jupyter = False

class ParallelSensitivityAnalysis(SensitivityAnalysis):
    def __init__(self, population=None, X=None, y=None, save=True, save_format='png', save_txt=True, verbose=True,
                 jupyter=False):
        SensitivityAnalysis.__init__(self, population=population, X=X, y=y, save=save,
                                     save_format=save_format, save_txt=save_txt,
                                     verbose=verbose, jupyter=jupyter)
        self.txt_list = [] if save_txt else None
        self.buckets = {}

    def _configure(self, config_file_path, x0_str, input_str, output_str, indep_norm, dep_norm, beta, rel_start,
                   p_baseline, r_ceiling_val, confound_baseline, global_log_indep, global_log_dep, repeat, n_neighbors,
                   max_neighbors, uniform):
        if self.rank == 0:
            SensitivityAnalysis._configure(
                self, config_file_path, x0_str, input_str, output_str, indep_norm, dep_norm,
                beta, rel_start, p_baseline, r_ceiling_val, confound_baseline, global_log_indep,
                global_log_dep, repeat, n_neighbors, max_neighbors, uniform, False)
        else:
            self.confound_baseline, self.p_baseline, self.r_ceiling_val = \
                confound_baseline, p_baseline, r_ceiling_val
            self.repeat, self.beta, self.rel_start = repeat, beta, rel_start
            self.n_neighbors = n_neighbors

    def _get_settings_from_root(self, comm):
        self.x0_str = comm.bcast(self.x0_str, root=0)
        self.input_str = comm.bcast(self.input_str, root=0)
        self.output_str = comm.bcast(self.output_str, root=0)
        self.indep_norm = comm.bcast(self.indep_norm, root=0)
        self.dep_norm = comm.bcast(self.dep_norm, root=0)

        self.global_log_indep = comm.bcast(self.global_log_indep, root=0)
        self.global_log_dep = comm.bcast(self.global_log_dep, root=0)

        self.input_names = comm.bcast(self.input_names, root=0)
        self.y_names = comm.bcast(self.y_names, root=0)
        self.inp_out_same = comm.bcast(self.inp_out_same, root=0)

    def _neighbor_search(self, max_neighbors, beta, X_x0_normed, n_neighbors, p_baseline, confound_baseline,
                         rel_start, repeat, uniform):
        #intermediate: list of list of neighbors
        neighbors_per_query = first_pass_p(
            self.X_normed, self.input_names, max_neighbors, beta, self.x0_idx,
            self.txt_list, self.buckets[self.rank])
        #intermediates: dict (input index : list of lists)
        neighbor_dict, confound_dict = clean_up_p(
            neighbors_per_query, self.X_normed, self.y_normed, X_x0_normed,
            self.input_names, self.y_names, n_neighbors, p_baseline, confound_baseline,
            rel_start, repeat, self.txt_list, self.verbose, uniform)
        return neighbors_per_query, neighbor_dict, confound_dict

    def _create_buckets(self, comm_size):
        """naiive"""
        min_elems = int(self.X.shape[1] / comm_size)
        num_bucket_with_max = self.X.shape[1] % comm_size
        counter = 0
        for i in range(comm_size):
            if i < num_bucket_with_max:
                self.buckets[i] = list(range(counter, counter + min_elems + 1))
                counter += min_elems + 1
            else:
                self.buckets[i] = list(range(counter, counter + min_elems))
                counter += min_elems

    def _merge(self, neighbors_per_query, neighbor_matrix, confound_matrix):
        """
        :param neighbors_per_query: a list (bc of comm.gather()) of dicts. key = input idx, val = list of lists
            where the number of lists = number of outputs and the elements = neighbor indices (ints)
        :param neighbor_matrix: same
        :param confound_matrix: same but the elements = input indices (also ints)
        :return: reformatted variables
        """
        new_neighbor_matrix = np.empty((self.X.shape[1], self.y.shape[1]), dtype=object)
        new_confound_matrix = np.empty((self.X.shape[1], self.y.shape[1]), dtype=object)
        new_neighbors_per_query = [[] for _ in range(self.X.shape[1])]

        for job in neighbor_matrix:
            for input_idx in job.keys():
                new_neighbor_matrix[input_idx] = job[input_idx]
        for job in confound_matrix:
            for input_idx in job.keys():
                new_confound_matrix[input_idx] = job[input_idx]
        for job in neighbors_per_query:
            for input_idx in job.keys():
                new_neighbors_per_query[input_idx] = job[input_idx]

        if self.save_txt:
            tmp_list = [elem for li in self.txt_list for elem in li]  # flatten 2d list
            self.txt_list = tmp_list

        return new_neighbors_per_query, new_neighbor_matrix, new_confound_matrix

    def _save_txt_file(self, save_path=None):
        if save_path is None: save_path = "data/lsa/{}{}{}{}{}{}_output_txt.txt".format(*time.localtime())
        txt_file = io.open(save_path, "w", encoding='utf-8')
        write_settings_to_file(
            self.input_str, self.output_str, self.x0_str, self.indep_norm, self.dep_norm,
            self.global_log_indep, self.global_log_dep, self.beta, self.rel_start,
            self.confound_baseline, self.p_baseline, self.repeat, txt_file)
        for line in self.txt_list:
            txt_file.write(line)
            txt_file.write("\n")
        txt_file.close()

    def _save_fp_colormaps(self, neighbor_arr, confound_matrix):
        pdf = PdfPages("data/lsa/{}{}{}{}{}{}_first_pass_colormaps.pdf".format(*time.localtime()))
        for i in range(self.X.shape[1]):
            plot_first_pass_colormap(neighbor_arr[i], self.X, self.y, self.input_names, self.y_names,
                                     self.input_names[i], confound_matrix[i], self.p_baseline,
                                     self.r_ceiling_val, pdf, self.save)

        pdf.close()

    def run_analysis(self, config_file_path=None, x0_idx=None, x0_str=None, input_str=None,
                     output_str=None, indep_norm=None, dep_norm=None, n_neighbors=60, max_neighbors=np.inf,
                     beta=2., rel_start=.5, p_baseline=.05, confound_baseline=.5, r_ceiling_val=None,
                     global_log_indep=None, global_log_dep=None, repeat=False, uniform=False):
        """
        :param config_file_path: str or None. path to yaml file, used to check parameter bounds on the perturbation vector
            (if the IV is the parameters). if config_file_path is not supplied, it is assumed that potentially generating
            parameter values outside their optimization bounds is acceptable.
        :param x0_idx: int or None (default). index of the center in the X array/PopulationStorage object
        :param x0_str: string or None. specify either x0_idx or x0_string, but not both. if both are None, a random
            center is selected. x0_string represents the center point of the neighbor search. accepted strings are 'best' or
            any of the objective names
        :param input_str: string representing the independent variable. accepted strings are 'parameter', 'p,' objective,'
            'o,' 'feature,' 'f.'
        :param output_str: string representing the independent variable. accepted strings are 'objective,'
            'o,' 'feature,' 'f.'
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
        :param global_log_indep: string or None. if indep_norm is 'loglin,' user can specify if normalization should be
            global or local. accepted strings are 'local' or 'global.'
        :param global_log_dep: string or None. if dep_norm is 'loglin,' user can specify if normalization should be
            global or local. accepted strings are 'local' or 'global.'
        :param repeat: Bool; if true, repeatedly checks the set of points to see if there are still confounds.
        :param uniform: bool; if True, will select a set of n_neighbor points after the clean up process that are as uniformly
            spaced as possible (wrt the query parameter)
        :return: PopulationStorage and LSA object. The PopulationStorage contains the perturbations. The LSA object is
            for plotting and saving results of the optimization and/or sensitivity analysis.
        """
        if self.lsa_completed:
            # gini is completely redone but it's quick
            plot_gini(
                self.X_normed, self.y_normed, self.input_names, self.y_names,
                self.inp_out_same, uniform, n_neighbors)
            InteractivePlot(
                self.plot_obj, searched=True, sa_obj=self, p_baseline=self.p_baseline,
                r_ceiling_val=self.r_ceiling_val)
            return self.plot_obj, self.perturb
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self._configure(
            config_file_path, x0_str, input_str, output_str, indep_norm, dep_norm, beta, rel_start,
            p_baseline, r_ceiling_val, confound_baseline, global_log_indep, global_log_dep, repeat,
            n_neighbors, max_neighbors, uniform)
        self._get_settings_from_root(comm)
        self._normalize_data(x0_idx)
        X_x0_normed = self.X_normed[self.x0_idx]

        if self.rank == 0:
            plot_gini(
                self.X_normed, self.y_normed, self.input_names, self.y_names,
                self.inp_out_same, uniform, n_neighbors)

        self._create_buckets(comm.Get_size())

        neighbors_per_query, neighbor_matrix, confound_matrix = self._neighbor_search(
            max_neighbors, beta, X_x0_normed, n_neighbors, p_baseline, confound_baseline,
            rel_start, repeat, uniform)

        # rank 0 waits and gathers intermediates
        neighbors_per_query = comm.gather(neighbors_per_query, root=0)
        neighbor_matrix = comm.gather(neighbor_matrix, root=0)
        confound_matrix = comm.gather(confound_matrix, root=0)
        self.txt_list = comm.gather(self.txt_list, root=0)

        if self.rank == 0:
            neighbors_per_query, neighbor_matrix, confound_matrix = self._merge(
                neighbors_per_query, neighbor_matrix, confound_matrix)

            self._plot_neighbor_sets(neighbors_per_query, neighbor_matrix, confound_matrix)
            coef_matrix, pval_matrix, failed_matrix = \
                self._compute_values_for_final_plot(neighbor_matrix)

            self.plot_obj = SensitivityPlots(
                pop=self.population, neighbor_matrix=neighbor_matrix,
                query_neighbors=neighbors_per_query, input_id2name=self.input_names,
                y_id2name=self.y_names, X=self.X_normed, y=self.y_normed, x0_idx=self.x0_idx,
                y_norm_obj=self.y_norm_obj, n_neighbors=n_neighbors,
                confound_matrix=confound_matrix, lsa_heatmap_values=self.lsa_heatmap_values,
                coef_matrix=coef_matrix, pval_matrix=pval_matrix, failed_matrix=failed_matrix)

            if self.input_str not in self.param_strings and self.population is not None:
                print("The parameter perturbation object was not generated because the "
                      "independent variables were features or objectives, not parameters.")
            else:
                self.perturb = Perturbations(
                    config_file_path, n_neighbors, self.population.param_names,
                    self.population.feature_names, self.population.objective_names, self.X,
                    self.x0_idx, neighbor_matrix)
            if self.save_txt:
                self._save_txt_file()
            if not self.jupyter:
                self._save_fp_colormaps(neighbors_per_query, confound_matrix)

            InteractivePlot(
                self.plot_obj, searched=True, sa_obj=self, p_baseline=p_baseline, r_ceiling_val=r_ceiling_val)
            self.lsa_completed = True

            plot_path = "data/{}{}{}{}{}{}_plot_object.pkl".format(*time.localtime())
            save(plot_path, self.plot_obj)
            print("Plot object saved to %s." % plot_path)
            perturb_path = "data/{}{}{}{}{}{}_perturb_object.pkl".format(*time.localtime())
            save(perturb_path, self.perturb)
            print("Perturbation object saved to %s." % perturb_path)
            self.save_analysis()


def first_pass_p(X, input_names, max_neighbors, beta, x0_idx, txt_list, bucket):
    neighbor_arr = {}
    x0_normed = X[x0_idx]
    X_dists = np.abs(X - x0_normed)
    output_text_p(
        "First pass: ",
        txt_list,
        True,
    )
    for input_idx in bucket:
        neighbors = []
        unimp = [x for x in range(X.shape[1]) if x != input_idx]  # non-query
        sorted_idx = X_dists[:, input_idx].argsort()  # sort by query

        for j in range(X.shape[0]):
            curr = X_dists[sorted_idx[j]]
            rad = curr[input_idx]
            if np.all(np.abs(curr[unimp]) <= beta * rad):
                neighbors.append(sorted_idx[j])
            if len(neighbors) >= max_neighbors: break
        neighbor_arr[input_idx] = neighbors
        max_dist = np.max(X_dists[neighbors][:, input_idx])
        output_text_p(
            "    %s - %d neighbors found. Max query distance of %.8f." %
                (input_names[input_idx], len(neighbors), max_dist),
            txt_list,
            True,
        )
    return neighbor_arr


def clean_up_p(neighbor_arr, X, y, X_x0, input_names, y_names, n_neighbors, p_baseline,
             confound_baseline, rel_start, repeat, txt_list, verbose, uniform):
    from diversipy import psa_select

    total_input = X.shape[1]
    neighbor_matrix = {}
    confound_matrix = {}
    for input_idx in neighbor_arr:
        nq = [x for x in range(total_input) if x != input_idx]
        confound_list = [[] for _ in range(y.shape[1])]
        neighbor_list = [[] for _ in range(y.shape[1])]

        for o in range(y.shape[1]):
            neighbors = neighbor_arr[input_idx].copy()
            counter = 0
            current_confounds = None
            rel = rel_start
            while current_confounds is None \
                    or (rel > 0 and len(current_confounds) != 0 and len(neighbors) > n_neighbors):
                current_confounds = []
                rmv_list = []
                for i2 in nq:
                    r = abs(linregress(X[neighbors][:, i2], y[neighbors][:, o])[2])
                    pval = linregress(X[neighbors][:, i2], y[neighbors][:, o])[3]
                    if r >= confound_baseline and pval < p_baseline:
                        output_text_p(
                            "Iteration %d: For the set of neighbors associated with %s vs %s, %s was "
                                "significantly correlated with %s." %
                                (counter, input_names[input_idx], y_names[o], input_names[i2], y_names[o]),
                            txt_list,
                            verbose,
                        )
                        current_confounds.append(i2)
                        for n in neighbors:
                            if abs(X[n, i2] - X_x0[i2]) > rel * abs(X[n, input_idx] - X_x0[input_idx]):
                                if n not in rmv_list: rmv_list.append(n)
                for n in rmv_list:
                    neighbors.remove(n)
                output_text_p(
                    "During iteration %d, for the pair %s vs %s, %d points were removed. %d remain." \
                        % (counter, input_names[input_idx], y_names[o], len(rmv_list), len(neighbors)),
                    txt_list,
                    verbose,
                )
                if not repeat:
                    break
                rel -= (rel_start / 10.)
                counter += 1
            confound_list[o] = current_confounds
            if repeat and len(current_confounds) != 0:
                neighbor_list[o] = []
            else:
                cleaned_selection = X[neighbors][:, input_idx].reshape(-1, 1)
                if uniform and len(neighbors) >= n_neighbors \
                        and np.min(cleaned_selection) != np.max(cleaned_selection):
                    renormed = (cleaned_selection - np.min(cleaned_selection)) \
                               / (np.max(cleaned_selection) - np.min(cleaned_selection))
                    subset = psa_select(renormed, n_neighbors)
                    idx_nested = get_idx(renormed, subset)
                    neighbor_list[o] =  np.array(neighbors)[idx_nested]
                else:
                    neighbor_list[o] = neighbors
            if len(neighbors) < n_neighbors:
                output_text_p(
                    "----Clean up: %s vs %s - %d neighbor(s) remaining!" \
                      % (input_names[input_idx], y_names[o], len(neighbors)),
                    txt_list,
                    True,
                )
        neighbor_matrix[input_idx] = neighbor_list
        confound_matrix[input_idx] = confound_list
    return neighbor_matrix, confound_matrix


def output_text_p(text, text_list, verbose):
    if verbose: print(text)
    if text_list is not None:
        text_list.append(text)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0:
    storage = PopulationStorage(file_path=storage_file_path)
else:
    storage = None
storage = comm.bcast(storage, root=0)
sa = ParallelSensitivityAnalysis(
    population=storage, save=save_imgs, save_format=save_format, save_txt=save_txt,
    verbose=verbose, jupyter=jupyter)
sa.run_analysis()