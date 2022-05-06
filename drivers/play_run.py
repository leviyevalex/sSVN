#%%
# import autograd.numpy as np  # Thinly-wrapped numpy
# from autograd import grad, hessian
# from source.stein_experiments import SVI
# from stein_thinning.thinning import thin
import numpy as np
# import os
# from os.path import join, dirname
# import h5py
# import matplotlib.pyplot as plt
# import deepdish as dd
# import corner
# import seaborn as sns
# from scipy import stats
import itertools
import copy
# from models.rosenbrock_analytic_proper_sample import rosenbrock_analytic
# x_in = np.array([0. , 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4])
#%%
# Hybrid Rosenbrock summation part
mu = 1.1
a = 1.2
n1 = 8
n2 = 3
dim = (n1-1) * n2 + 1
x_in = np.random.rand(dim)

def hybrid_rosenbrock_vectorized(n2, n1, x_in):
    x_graph = np.insert(x_in[1:].reshape(n2, n1-1), 0, x_in[0], axis=1)
    b = np.ones((n2, n1-1))
    term_a = a * (x_in[0] - mu) ** 2
    term_b = np.sum(b * (x_graph[:, 1:] - x_graph[:, :-1] ** 2) ** 2)
    return term_a + term_b

def hybrid_rosenbrock_manual(n2, n1, x_in):
    x_graph = np.insert(x_in[1:].reshape(n2, n1-1), 0, x_in[0], axis=1)
    rangej = range(n2)
    rangei = np.arange(1, n1)
    term_a = a * (x_in[0] - mu) ** 2
    term_b = 0
    for j, i in itertools.product(rangej, rangei):
        val = 1 * (x_graph[j, i] - x_graph[j, i-1] ** 2) ** 2
        term_b += val
    return term_a + term_b

assert np.allclose(hybrid_rosenbrock_manual(n2,n1,x_in), hybrid_rosenbrock_vectorized(n2, n1, x_in), 1e-6)


#%%

######################################################
# The n2=1, n1=2 case
######################################################
n1 = 2
n2 = 1
dim = (n1-1) * n2 + 1
x = np.random.rand(dim)
b = np.random.rand(n2, n1)
test_a = b[0,1] * (x[0,1] - x[0,0] ** 2) ** 2
#%%
x_a = x[1:].reshape(n2, n1-1) # Perform trimming
x_b = x[]
#%%
test_b = 0
rangej = range(n2)
rangei = np.arange(1, n1)
for j, i in itertools.product(rangej, rangei):
    print(j,i)
    val = b[j,i] * (x[j, i] - x[j, i-1] ** 2) ** 2
    test_b += val
#%%
x_ = copy.copy(x)
x_trim = b[:, -1:] * (x_[:,-1:] - x_[:,0:n2] ** 2) ** 2
#####################################################
# Harder general case
#####################################################
#%%
n1 = 4
n2 = 2
dim = (n1-1) * n2 + 1
x = np.random.rand(dim)
x_trim = copy.copy(x[1:]).reshape(n2, n1-1)
b = np.random.rand(n2, n1)
test_b = 0
rangej = np.arange(0, n2)
rangei = np.arange(1, n1-1)
#%%
for j, i in itertools.product(rangej, rangei):
    print(j,i)
    val = b[j,i] * (x_trim[j, i] - x_trim[j, i-1] ** 2) ** 2
    test_b += val
#%%
x_test = b[:, -1:] * (x_[:,-1:] - x_[:,0:n2] ** 2) ** 2
#%%


#%%
x_trim = copy.deepcopy(x_[1:]).reshape(n2, n1-1)
#%%
x_graph[:,1:] -= x_graph[:,:-1] ** 2
test_c = np.einsum('ji, ji ->', b, x_graph ** 2)
#%%
test_a = b * (x_trim[])



# a = 1.2
x_graph = copy.deepcopy(x_.reshape(n2, n1-1))
#%%
# b = np.random.rand(n2, n1 - 1)
# y = np.zeros((n2, n1 - 1))
# r = 0
#%%
res = 0
shape1 = x_graph.shape[0]
shape2 = x_graph.shape[1]
jrange = np.arange(0, n2)
irange = np.arange(1, n1)
#%%
for j, i in itertools.product(range(n2), range(n1-1)):
    print('b : %f' % b[j, i])
    print('x_graph: %f', x_graph[j, i])
    print('x_graph_lag: %f', x_graph[j, i-1])
    val = b[j,i] * (x_graph[j, i] - x_graph[j, i-1] ** 2) ** 2
    res += val
#%%
x_graph_copy = copy.deepcopy(x_graph)
test = copy.deepcopy(b * (x_graph[:,1:] - x_graph[:,:-1] ** 2) ** 2)



#%%
x_graph[:,1:] -= x_graph[:,:-1] ** 2
test1 = np.einsum('ji, ji ->', b, x_graph ** 2)
#%%
#%%
 # pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)