import gmmoments as gmm

import numpy as np
import numpy.random as random

# I need a test application
# for that I need some data
# how about an IV application?
# y ~ b0 + b1 * x1 + b2 * x2 + c0 * q0 + eps (q is unobserved)
# x1 ~ N(0, sig)
# x2 = a + b * q0 + d0 * z + eps (z is observed)
# z ~ N(0, sig)

def generate_data(ni, sigma2, coefs_y, coefs_x2):
    eps_y, x1, eps_x2, q, z = np.transpose(random.normal(size=(ni, 5)) * np.sqrt(sigma2)) # ey, ex1, ex2, eq, ez
    x2 = coefs_x2[0] + coefs_x2[1] * q + coefs_x2[2] * z + eps_x2
    y = coefs_y[0] + coefs_y[1] * x1 + coefs_y[2] * x2 + coefs_y[3] * q + eps_y
    return np.vstack((y, x1, x2, z))

def g(di, theta): # the moment function
    y, x1, x2, z = di
    resid = y - (theta[0] + theta[1] * x1 + theta[2] * x2)
    return np.array([resid, resid * x1, resid * z])

##
def test_estimate():
    ni = 1_000
    nsamples = 1_000

    sigma2 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    coefs_y = np.array([1.0, 0.5, -2.0, 0.8])
    coefs_x2 = np.array([-3.0, 1.0, 1.5])
    target = np.array([1.0, 0.5, -2.0])
    init = np.ones(3)

    ests = np.array([gmm.estimate(g, generate_data(ni, sigma2, coefs_y, coefs_x2), init) for _ in range(nsamples)])
    err = np.mean(np.square(np.mean(ests, axis=0) - target))
    assert err < 0.1
