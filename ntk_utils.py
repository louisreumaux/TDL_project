import matplotlib.pyplot as plt
import functools
import jax.numpy as jnp


def loss_fn(predict_fn, z_targets, ts, *args):
    def loss_fn_single(t):
        predictions = predict_fn(t, *args).ntk
        return jnp.mean((predictions - z_targets) ** 2)
    return jnp.array([loss_fn_single(t) for t in ts])

def format_plot(x=None, y=None):
  
  ax = plt.gca()
  if x is not None:
    plt.xlabel(x, fontsize=20)
  if y is not None:
    plt.ylabel(y, fontsize=20)

