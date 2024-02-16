import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def get_data():
    np.random.seed(0)
    X = np.random.rand(2**12, 1)
    Y = np.random.rand(2**12, 1)


    input_training = np.column_stack((X, Y))
    z_training  = 100 * (X - 0.2) * (X - 0.1) * (X - 0.5) * (X - 1) + (Y - 0.5) + 3
    noise = np.random.rand(z_training.shape[0], z_training.shape[1])
    z_training += 0.1*noise

    X_test = np.random.rand(2**9, 1)
    Y_test = np.random.rand(2**9, 1)

    input_test = np.column_stack((X_test, Y_test))
    z_test = 100 * (X_test - 0.2) * (X_test - 0.1) * (X_test - 0.5) * (X_test - 1) + (Y_test - 0.5) + 3

    return input_training, z_training, input_test, z_test

def log_scaling_law(n, log_A, beta_log, alpha):
    return log_A + beta_log * np.log(1/n + alpha)


def get_beta_scaling_laws(min_list_errors, n_samples_list):
    
    log_errors = np.log(min_list_errors)
    popt, pcov = curve_fit(log_scaling_law, n_samples_list, log_errors, maxfev=100000)
    log_A_fit, beta_log_fit, alpha = popt
    A_fit = np.exp(log_A_fit)

    return beta_log_fit

def plot_nn_vs_ntk(n_samples_list, test_error_ntk, test_error_nn):

    plt.plot(n_samples_list, test_error_ntk, label='Infinite NTK')
    plt.plot(n_samples_list, test_error_nn, label='Neural Network')

    plt.xscale('log', base=2)

    plt.title('NTK at initialization')

    plt.xlabel('Train size')
    plt.ylabel('Test error')

    plt.legend()
    plt.show()