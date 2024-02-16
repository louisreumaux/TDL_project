from jax import jit, grad, vmap, random
import neural_tangents as nt
from neural_tangents import stax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import functools
from ntk_utils import loss_fn, format_plot

legend = functools.partial(plt.legend, fontsize=10)

def ntk_computation(n_samples_list, input_training, z_training, input_test, z_test):

    key = random.PRNGKey(0)
    key, subkey = random.split(key)

    min_list_ntk = []

    for nb_samples in n_samples_list:

        input_sample = input_training[:nb_samples]
        z_sample = z_training[:nb_samples]

        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(512), stax.Erf(),
            stax.Dense(512), stax.Erf(),
            stax.Dense(1)
        )

        predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, input_sample, z_sample)

        ts = jnp.arange(0, 2*10 ** 7, 5*10**5)


        ntk_test_loss_mean = loss_fn(predict_fn, z_test, ts, input_test)
        print(np.min(ntk_test_loss_mean))

        min_list_ntk.append(np.min(ntk_test_loss_mean))

    return min_list_ntk