from ntk import ntk_computation
from utils import get_data, get_beta_scaling_laws, plot_nn_vs_ntk
from nn_training import nn_training_sizes


n_samples_list = [2**x for x in range(3,10)]

input_training, z_training, input_test, z_test = get_data()
test_error_ntk = ntk_computation(n_samples_list, input_training, z_training, input_test, z_test)
test_error_nn = nn_training_sizes(n_samples_list, input_training, z_training, input_test, z_test)

beta_nn = get_beta_scaling_laws(test_error_nn, n_samples_list)
beta_ntk = get_beta_scaling_laws(test_error_ntk, n_samples_list)

plot_nn_vs_ntk(n_samples_list, test_error_ntk, test_error_nn)

print(f"Scaling exponent beta of {beta_nn} for neural network")
print(f"Scaling exponent beta of {beta_ntk} for infinite NTK")


