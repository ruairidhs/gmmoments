import jax
jax.config.update("jax_enable_x64", True) # use float64
from jax import numpy as jnp
from jax.scipy import optimize as jopt
from functools import partial # for @jax.jit

def estimate(g, data, init_guess):
    W = jnp.identity(get_n_moments(g, data, init_guess))
    first_step = _find_min(g, data, W, init_guess)
    W2 = estimate_opt_weighting_matrix(g, data, first_step)
    est = _find_min(g, data, W2, first_step)
    W3 = estimate_opt_weighting_matrix(g, data, est)
    vcov = compute_std_errs(g, data, W3, est)
    serrs = jnp.sqrt(jnp.diag(vcov))
    return {'coef': est, 'stderr': serrs, 'vcov': vcov}

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

@partial(jax.jit, static_argnums=(0,))
def estimate_opt_weighting_matrix(g, data, theta):
    n = data.shape[0]
    g_individual = jax.vmap(g, in_axes=(0, None))(data, theta)
    Omega = (1 / n) * (jnp.transpose(g_individual) @ g_individual)
    return jnp.linalg.inv(Omega)

@partial(jax.jit, static_argnums=(0,))
def compute_std_errs(g, data, W, theta):
    n = data.shape[0]
    J = jax.jacfwd(lambda theta: jnp.mean(jax.vmap(g, in_axes=(0, None))(data, theta), axis=0))(theta)
    vcov = (1 / n) * jnp.linalg.inv(jnp.transpose(J) @ W @ J)
    return vcov
