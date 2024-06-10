import jax
jax.config.update("jax_enable_x64", True) # use float64
from jax import numpy as jnp
from jax.scipy import optimize as jopt
from functools import partial # for @jax.jit

def estimate(g, data, init_guess):
    W = jnp.identity(get_n_moments(g, data, init_guess))
    return _find_min(g, data, W, init_guess)

@partial(jax.jit, static_argnums=(0,)) # critical for performance!
def _find_min(g, data, W, init):
    obj = make_objective_function(g, data, W)
    return jopt.minimize(obj, init, method='BFGS').x

def get_n_moments(g, data, init_guess):
    di = data[0, :]
    out = g(di, init_guess)
    return len(out)

def make_objective_function(g, data, W):
    def obj(theta):
        gsample = jnp.mean(jax.vmap(g, in_axes=(0, None))(data, theta), axis=0)
        return gsample @ W @ jnp.transpose(gsample)
    return obj

# And now I want the two-stage and standard errors
