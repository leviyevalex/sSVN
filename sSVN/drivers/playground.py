#%%
# import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad, hessian
from source.stein_experiments import SVI
from stein_thinning.thinning import thin
import numpy as np
import os
from os.path import join, dirname
import h5py
import matplotlib.pyplot as plt
import deepdish as dd
import corner
import seaborn as sns
from scipy import stats
import itertools
from models.rosenbrock_analytic_proper_sample import rosenbrock_analytic
#%%
n1 = 4
n2 = 5
a = 1.2
dim = (n1-1) * n2 + 1
x = np.random.rand(dim)
x_ = x[1:]
x_graph = x_.reshape(n2, n1-1)
#%%
# b = np.random.rand(n2, n1 - 1)
# y = np.zeros((n2, n1 - 1))
r = 0
res = 0
#%%
for j, i in itertools.product(range(n2), range(n1-1)):
    # print(i, j)
    val = (x_graph[j, i] - x_graph[j, i-1])
    # print(val)
    res = res + val
    print(val)
    print(r)
#%%
np.einsum('ji ->', x_graph[:,1:] - x_graph[:,:-1])
#%%
res =
#%%



#%%
A = np.array([[2, 0],
              [0, 2]])
B = np.array([[1, 2, 3],
              [1, 2, 3]])
# B = np.random.normal(0, 1, (2, 5))
#%%
C = A @ B




#%%


def check_if_same(A, B, tol=None):
    if tol==None:
        # tol = 1e-16
        tol = 1e-25
    return np.allclose(A, B, rtol=tol, atol=tol)
#%%
data1 = dd.io.load('DEBUG_ACCURACY_81.h5')
data2 = dd.io.load('DEBUG_ACCURACY_87.h5')
#%%
print(check_if_same(data1['kx'], data2['kx']))
print(check_if_same(data1['gkx'], data2['gkx']))
print(check_if_same(data1['hesskx'], data2['hesskx']))
print(check_if_same(data1['xi'], data2['xi']))
print(check_if_same(data1['a'], data2['a']))
print(check_if_same(data1['b'], data2['b']))
print(check_if_same(data1['c'], data2['c']))
#%%



#%%
a1d = dd.io.load('81_iterations.h5')
a2d = dd.io.load('87_iterations.h5')
#%%
a1 = dd.io.load('81_iterations_tpqo.h5')
a2 = dd.io.load('87_iterations_tpqo.h5')

#%%
print(check_if_same(a1d[0]['h'], a2d[0]['h'], 1e-25))
print(check_if_same(a1d[0]['h_med'], a2d[0]['h_med'], 1e-25))
#%%
print(check_if_same(a1[0]['cost_dict0']['h'], a2[0]['cost_dict0']['h']))
print(check_if_same(a1[0]['cost_dict1']['h'], a2[0]['cost_dict1']['h']))
print(check_if_same(a1[0]['cost_dict2']['h'], a2[0]['cost_dict2']['h']))
print(check_if_same(a1[0]['gCost0'], a2[0]['gCost0']))
print(check_if_same(a1[0]['s'], a2[0]['s']))
print(check_if_same(a1[0]['cost0eps'], a2[0]['cost0eps']))
#%%
#%%
cwd = os.getcwd()
# C:\Users\Alex\Projects\stein_variational_newton\drivers\outdir\1613484888
metadata = '/drivers/outdir/1613485405/hybrid_rosenbrock_nP_1000_150_1613485405.h5' # 5d rosen
file = cwd + metadata
results = dd.io.load(file)

#%%
# Stein thinning experiment
fig_thin, ax_thin = plt.subplots(figsize = (10, 10))
cp = ax_thin.contour(X, Y, Z, 7, colors='black', alpha=0.1)
ax_thin.set_facecolor('#F5FEFF')
fig_thin.patch.set_facecolor('#F5FEFF')
model = rosenbrock_analytic()
logger_output_path = 'info.log'
stein = SVI(model = model, log_output_path=logger_output_path, nIterations=1, nParticles=1)
grad_results = -1 * stein.getGradientMinusLogPosterior_ensemble_new(results)
idx = thin(results, grad_results, 1000)
idx_unique = np.unique(idx)
ax_thin.scatter(results[:,0], results[:, 1], marker=".", color='#51AEFF', s=8)
ax_thin.scatter(results[idx_unique][:,0], results[idx_unique][:, 1], marker=".", color='r', s=8)


#%%
cwd = os.getcwd()
# C:\Users\Alex\Projects\stein_variational_newton\drivers\outdir\1613484888
folder = '/1613269546 HE this one is awesome'
filename = '/rosenbrock_proper_nP_1000_200_1613269546.h5'
metadata = '/drivers/outdir' + folder + filename # 2d rosen good
file = cwd + metadata
results = dd.io.load(file)[200]['X']
ground_truth_path = cwd + '/models/ground_truths/double_rosenbrock/3k_samples_rosen_test.h5'
ground_truth = dd.io.load(ground_truth_path)
# ground_truth_path = cwd + '/models/ground_truths/double_rosenbrock/170k.h5'
# ground_truth = dd.io.load(ground_truth_path).T
#%%
# background = '/drivers/outdir/' + 'hybrid_rosenbrock_-2.50_2.50.h5'
background = '/drivers/outdir/' + 'rosenbrock_proper_-2.50_2.50.h5'
background_path = cwd + background
background_data = dd.io.load(background_path)
X = background_data['X']
Y = background_data['Y']
Z = background_data['Z']
#%%
fig, ax = plt.subplots(figsize = (10, 10))
cp = ax.contour(X, Y, Z, 7, colors='black', alpha=0.1)
ax.set_facecolor('#F5FEFF')
fig.patch.set_facecolor('#F5FEFF')
ax.scatter(results[:,0], results[:, 1], marker=".", color='#51AEFF', s=8)
plt.axis('off')
#%%
fig.show()
fig, ax = plt.subplots()
plt.contourf(X,Y,Z,7., cmap='Blues', alpha=0.5)
# plt.colormap('white')
#%%

# hold;
# Xp = Xout1(:,1);
# Yp = Xout1(:,2);
# hp = scatter(Xp,Yp,40,'filled');
# set(gcf,'position',[40,40,640,640]);
# alpha(hp,0.9);
# title('MCMC','FontSize',64);
#%%
plt.style.use('seaborn-dark-palette')
plt.style.use('seaborn-dark')
fig, ax = plt.subplots()
#%%
gauss_kde = stats.gaussian_kde(results.T)
gauss_kde.set_bandwidth(.07)
resample = gauss_kde.resample(1000).T
#%%
plt.scatter(resample[:,0], resample[:, 1], c='r')
plt.scatter(results[:,0], results[:, 1], c='b')
#%%
# sns.kdeplot(results, bw=.07)
fig.show()
#%%
# PP PLOT
percs = np.linspace(0,100,21)
for d in range(results.shape[1]):
    qn_a = np.percentile(ground_truth[:, d], percs)
    qn_b = np.percentile(results[:, d], percs)
    label = 'Dimension %i' % d
    ax.plot(qn_a,qn_b, ls="", marker="o", label=label)
#%%
x = np.linspace(np.min((qn_a.min(),qn_b.min())), np.max((qn_a.max(),qn_b.max())))
ax.plot(x,x, color="k", ls="--")
ax.set_title("PP Plot")
ax.set_xlabel('Ground truth percentile')
ax.set_ylabel('SVGD percentile')
ax.legend()
fig.show()


#%%
percentiles = np.arange(1,100)
d = results[150]['X'][:,0]
res = np.percentile(d, percentiles)
plt.style.use('seaborn-dark-palette')
plt.style.use('seaborn-dark')
fig, ax = plt.subplots()
ax.set_title("PP Plot")
ax.set_xlabel('Percentile')
ax.set_ylabel('Percentile')
ax.plot(res, label='')

fig.show()
#%%
figure = corner.corner(results[150]['X'])
figure.show()
# f = h5py.File(file)
# d = f['/data']



#%%
def getMMD(label):
    def load_file_metadata(label):
        cwd = os.getcwd()
        label_str = str(label)
        metadata_path = cwd + '/drivers/outdir/' + label_str + '/metadata.h5'
        return dd.io.load(metadata_path)
    def extract_logmmd(metadata):
        last_iter = list(metadata.keys())[-2]
        mmd = np.array([])
        for l in range(last_iter):
            mmd_l = metadata[l]['MMD']
            mmd = np.append(mmd, mmd_l)
        return np.log(mmd)
    metadata = load_file_metadata(label)
    return extract_logmmd(metadata)

mmd_svgd_bmbw = getMMD(1606707878)
mmd_svgd_med = getMMD(1606347736)
mmd_svgd_linesearch = getMMD(1607040024)

plt.style.use('seaborn-dark-palette')
plt.style.use('seaborn-dark')
fig, ax = plt.subplots()
ax.set_title("MMD vs Iteration")
ax.set_xlabel('Iteration')
ax.set_ylabel('Log MMD')
ax.plot(mmd_svgd_bmbw, label='SVGD BMBW + eps=0.1')
ax.plot(mmd_svgd_med, label='SVGD MED')
ax.plot(mmd_svgd_linesearch, label='SVGD BMBW + linesearch')
plt.legend()
plt.show()
# results_svgd = load_file_metadata(1606347736)
# results_svn_I_BW = load_file_metadata(1606345192)
# results_svn_metric_BW = load_file_metadata(1606351626)
# results_svgd_BMBW = load_file_metadata(1606707878)
# results_SVN_BMBW = load_file_metadata(1606709532)
# results_SVGD_linesearch = load_file_metadata(1607032197)
# results_SVGD_constant_eps = load_file_metadata(1606853947)
# #%%
# mmd_svn_I_BW = extract_logmmd(results_svn_I_BW)
# mmd_svgd = extract_logmmd(results_svgd)
# mmd_svn_metric_BW = extract_logmmd(results_svn_metric_BW)
# mmd_svgd_BMBW = extract_logmmd(results_svgd_BMBW)
# mmd_SVN_BMBW = extract_logmmd(results_SVN_BMBW)
# mmd_SVGD_linesearch = extract_logmmd(results_SVGD_linesearch)
# mmd_SVGD_constant_eps = extract_logmmd(results_SVGD_constant_eps)
#%%
#%%
a = np.array([1, 1, 2, 2])
b = a.reshape(2, 2)

# ax.plot(mmd_svn_I_BW, label='SVN MED BW')
# ax.legend(['SVN MED BW'])
# ax.plot(mmd_svgd, label='SVGD MED BW')
# ax.legend(['SVGD MED BW'])
# ax.plot(mmd_svn_metric_BW, label='SVN METRIC BW')
# ax.legend(['SVN METRIC BW'])
# ax.plot(mmd_svgd_BMBW, label='SVGD BMBW')
# ax.plot(mmd_SVN_BMBW, label='SVN BMBW')
# ax.plot(mmd_SVGD_constant_eps, label='SVGD CONSTANT EPS')

# plt.style.use('fivethirtyeight')
#%%
#%%
#%%
#%%
# parallel stuff
# if rank == 0:
#     # in real code, this section might
#     # read in data parameters from a file
#     numData = 10
#     comm.send(numData, dest=1)
#
#     data = np.linspace(0.0,3.14,numData)
#     comm.Send(data, dest=1)
#
# elif rank == 1:
#
#     numData = comm.recv(source=0)
#     print('Number of data to receive: ',numData)
#
#     data = np.empty(numData, dtype='d')  # allocate space to receive the array
#     comm.Recv(data, source=0)
#
#     print('data received: ',data)


#%%
def l(x):
    eplus = np.exp(-2 * (x[0] + 3) ** 2)
    eminus = np.exp(-2 * (x[0] - 3) ** 2)
    pre = np.exp(-2 * (np.linalg.norm(x) - 3) ** 2)
    return pre * (eplus + eminus)

#%%
grad_l = grad(l)       # Obtain its gradient function
a = np.array([3.0, 0.])
b = grad_l(a)
hess_l = hessian(l)
c = hess_l(a)
#%%

# function [dlog_p] = dlog_p_toy(X_in)
# X = X_in;
# [N,d] = size(X);
# norm_X = sqrt(sum(X.^2,2));
# X1 = X(:,1);
# Z2 = zeros(N,1);
# expX11 = exp(-2*(X1-3).^2);
# expX12 = exp(-2*(X1+3).^2);
# dexpX1 = 4*((X1-3).*expX11+(X1+3).*expX12)./(expX11+expX12);
# dlog_p = -4*X./norm_X.*(norm_X-3)-[dexpX1,Z2];
#
# end
#%%
begin = -5
end = 5
def getMinusLogLikelihood_individual(x):
    eplus = np.exp(-2 * (x[0] + 3) ** 2)
    eminus = np.exp(-2 * (x[0] - 3) ** 2)
    nx = np.linalg.norm(x)
    pre = np.exp(-2 * (nx - 3) ** 2)
    tmp = pre * (eplus + eminus)
    return np.exp(-tmp)
def getMinusLogLikelihood_ensemble_new(thetas):
    # if thetas.shape[0] == self.DoF:
    #     thetas = thetas.T
    return np.apply_along_axis(getMinusLogLikelihood_individual, 1, thetas)
#%%
ngrid = 1000
# ngrid = 500
x = np.linspace(begin, end, ngrid)
y = np.linspace(begin, end, ngrid)
X, Y = np.meshgrid(x, y)
Z = np.exp(-1 * getMinusLogLikelihood_ensemble_new(np.vstack((np.ndarray.flatten(X), np.ndarray.flatten(Y))).T).reshape(ngrid,ngrid))
# Z = getMinusLogLikelihood_ensemble_new(np.vstack((np.ndarray.flatten(X), np.ndarray.flatten(Y))).T).reshape(ngrid,ngrid)
fig, ax = plt.subplots(figsize = (10, 10))
ax.set_xlabel('mass_1')
ax.set_ylabel('mass_2')
ax.set_title('SVN particle flow', fontdict={'fontsize': 20})
#%%
cp = ax.contourf(X, Y, Z, 10, cmap='viridis')
fig.colorbar(cp)
#%%
fig.show()




# #%%
# a = np.array([[2, 1],
#               [1, 2]])
# #%%
# def sum_diag(A):
#     return (A.sum() - np.diag(A).sum())/2
# #%%
# np.sum(np.triu(a))
# #%%
#
# # relative = '/models/ground_truths/double_rosenbrock/170k.h5'
# relative = '/models/ground_truths/double_rosenbrock/3k_samples_rosen_test.h5'
# cwd = os.getcwd()
# file = cwd + relative
# f = h5py.File(file)
# d = f['/data']
# da = da.from_array(d, chunks = 'auto')
#%%
def optimize_test(h0):
    hi0 = 1 / h0
    def cost(hi):
        h = 1 / hi
        try:
            return (h - 3) ** 2
        except:
            return np.infty
    explore_ratio = 1.1
    cost0 = cost(hi0)
    eps = 1e-6
    gCost0 = (cost(hi0 + eps) - cost0) / eps
    if gCost0 < 0:
        hi1 = hi0 * explore_ratio
    else:
        hi1 = hi0 / explore_ratio
    cost1 = cost(hi1)
    s = (cost1 - cost0) / (hi1 - hi0)
    hi2 = (hi0 * s - 0.5 * gCost0 * (hi1 + hi0)) / (s - gCost0)
    cost2 = cost(hi2)
    if hi2 != None and hi2 > 0:
        if cost1 < cost0:
            if cost2 < cost1:
                h = 1 / hi2
            else:
                h = 1 / hi1
        else:
            if cost2 < cost0:
                h = 1 / hi2
            else:
                h = h0
    else:
        if cost1 < cost0:
            h = 1 / hi1
        else:
            h = h0
    return h


# import argparse
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# # from source.stein_experiments import BILBY_TAYLORF2
# import inspect
#%%
import numdifftools as nd
# import bilby
# import random
# from cg_variants import hs_cg, cg_cg, gv_cg, pr_cg, pipe_pr_cg
# # from cg_variants import pipe_pr_cg
# # import PyTrilinos
# import scipy
# from autograd import value_and_grad
# from mpi4py import MPI
# import pydevd_pycharm
# # comm = MPI.COMM_WORLD
# # size = comm.Get_size()
# # rank = comm.Get_rank()
# # port_mapping=[50413, 54162]
# # pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)