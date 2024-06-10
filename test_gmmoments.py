import gmmoments as gmm

import numpy as np
import numpy.random as random

import jax
jax.config.update("jax_enable_x64", True) # use float64
from jax import numpy as jnp
from jax.scipy import optimize as jopt

##
# ---- IV Test Application ----
# y ~ b0 + b1 * x1 + b2 * x2 + c0 * q0 + eps (q is unobserved)
# x1 ~ N(0, sig)
# x2 = a + b * q0 + d0 * z + eps (z is observed)
# z ~ N(0, sig)
def generate_data(ni, sigma2, coefs_y, coefs_x2):
    eps_y, x1, eps_x2, q, z = np.transpose(random.normal(size=(ni, 5)) * np.sqrt(sigma2)) # ey, ex1, ex2, eq, ez
    x2 = coefs_x2[0] + coefs_x2[1] * q + coefs_x2[2] * z + eps_x2
    y = coefs_y[0] + coefs_y[1] * x1 + coefs_y[2] * x2 + coefs_y[3] * q + eps_y
    return np.transpose(np.vstack((y, x1, x2, z))) # (ni × 4)

@jax.jit
def g(di, theta): # the moment function
    y, x1, x2, z = di
    resid = y - (theta[0] + theta[1] * x1 + theta[2] * x2)
    return jnp.array([resid, resid * x1, resid * z])

##
def distance(u, v):
    return np.mean(np.square(u - v))

def test_estimate():
    ni = 1_000
    nsamples = 1_000

    sigma2 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    coefs_y = np.array([1.0, 0.5, -2.0, 0.8])
    coefs_x2 = np.array([-3.0, 1.0, 1.5])
    target = np.array([1.0, 0.5, -2.0])
    init = np.zeros(3)

    out = [gmm.estimate(g, generate_data(ni, sigma2, coefs_y, coefs_x2), init) for _ in range(nsamples)]
    ests = np.array([o['coef'] for o in out])
    ses = np.array([o['stderr'] for o in out])

    ests_err = distance(np.mean(ests, axis=0), target)
    ests_std = np.std(ests, axis=0)
    se_err = distance(ests_std, np.mean(ses, axis=0))
    assert ests_err < 1e-5
    assert se_err < 1e-5
