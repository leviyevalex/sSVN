import sys
import bilby
import matplotlib.pyplot as plt
#import pixiedust
# import sympy
import numpy as np
from matplotlib.pyplot import figure
from bilby import utils
#%%

import numpy as np
from scipy.sparse.linalg import cg
# import tensorflow as tf
import time



#%%
def set_to_variable_at_index(theta,index,m):
    theta[index] = float(m)
    return theta

def return_marginal_plot(model, inject, resolution, delta):
    """
    Plots the marginals for testing purposes
    Args:
        inject (ndarray OF FLOATS): Won't work if not a float ndarray. Used to plot region where marginal will be peaked
        resolution (float): How much to subdivide region. EX: .001
        delta (float): Window for plot

    Returns: matplotlib figure of marginals

    """
    figure, axes = plt.subplots(inject.size, 1)
    valcounter = 0
    for val in inject:
        inject_copy = np.copy(inject)

        # Only plot for positive values
        if val - delta < 0:
            domain = np.arange(0, val + delta, resolution)

        else:
            domain = np.arange(val - delta, val + delta, resolution)

        likelihood_range = np.ones(domain.size)

        mcounter = 0
        for m in np.nditer(domain):
            # Calculate the range
            set_to_variable_at_index(inject_copy, valcounter, m)
            likelihood_range[mcounter] = model.Bilby_BNS_Log_Likelihood_Function(inject_copy)
            mcounter += 1
        # Create subplots:
        for row in axes:
            x = domain
            y = likelihood_range

            axes[valcounter].set_xlabel("%s Guess" % model.injection_parameter_order[valcounter])
            axes[valcounter].set_ylabel("log Likelihood")
            axes[valcounter].set_title("%s Cross Section at Expected Peak" % model.injection_parameter_order[valcounter])
            #axes[valcounter].legend(loc='upper right')
            axes[valcounter].plot(x, y)

        valcounter += 1
    figure.tight_layout()
    figure.canvas.draw()
    return figure

pass

# -*- coding: utf-8 -*-



def conjugate_grad(A, b, x=None):
    """
    Description
    -----------
    Solve a linear equation Ax = b with conjugate gradient method.
    Parameters
    ----------
    A: 2d numpy.array of positive semi-definite (symmetric) matrix
    b: 1d numpy.array
    x: 1d numpy.array of initial point
    Returns
    -------
    1d numpy.array x such that Ax = b
    """
    n = len(b)
    if not x:
        x = np.ones(n)
    r = np.dot(A, x) - b
    p = - r
    r_k_norm = np.dot(r, r)
    for i in range(2*n):
        Ap = np.dot(A, p)
        alpha = r_k_norm / np.dot(p, Ap)
        x += alpha * p
        r += alpha * Ap
        r_kplus1_norm = np.dot(r, r)
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm
        if r_kplus1_norm < 1e-5:
            print('Itr: %i' % i)
            break
        p = beta * p - r
    return x

