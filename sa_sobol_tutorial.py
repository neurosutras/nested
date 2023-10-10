# running lsa
from nested.lsa import *
from nested.optimize_utils import *

pop = OptimizationHistory(file_path="data/20190930_1534_pa_opt_hist_test.hdf5")
sa = ParameterSensitivity(population=pop)
plot, perturb = sa.run_analysis()  # can set no_lsa=True

perturb.create()  # optional
plot.plot_vs_unfiltered("x1", "g")  # optional

# running sobol from command line:
# mpiexec -n 3 python -m nested.optimize --config-file-path="config/basic_example_config.yaml" --param_gen="Sobol" --sobol-analysis --num_models=100
