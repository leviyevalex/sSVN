import numpy as np
import os
# import dask.array as da
import h5py
import matplotlib.pyplot as plt
# import deepdish as dd
import os
import scipy
plt.rcParams['text.usetex'] = True
def create_moment_convergence_plots(output_dir):
    output_data_path = os.path.join(output_dir, 'output_data_new.h5')
    figures_directory = os.path.join(output_dir, 'figures')
    if os.path.isdir(figures_directory) == False:
        os.mkdir(figures_directory)

    with h5py.File(output_data_path, 'r') as hf:
        keys = [int(l) for l in list(hf.keys())[:-2]]
        keys.sort()
        iters_performed = keys[-1]
        mean_x = np.zeros(iters_performed)
        mean_y = np.zeros(iters_performed)
        mean = np.zeros(iters_performed)
        cov_x = np.zeros(iters_performed)
        cov_y = np.zeros(iters_performed)
        for l in range(iters_performed):
            X = hf['%i' % l]['X'][()]
            mean = np.mean(X, axis=0)
            # mean[l] = np.mean(X)
            cov = np.cov(X.T)
            mean_x[l] = mean[0]
            mean_y[l] = mean[1]
            cov_x[l] = cov[0, 0]
            cov_y[l] = cov[1, 1]
    plt.style.use('seaborn-dark-palette')
    plt.style.use('seaborn-dark')
    fig, ax = plt.subplots()
    ax.plot(mean_x, label='mean x')
    ax.plot(mean_y, label='mean y')
    ax.plot(cov_x, label='cov x')
    ax.plot(cov_y, label='cov y')
    # ax.axhline(y = 1, ls='--', c='black', lw=0.1)
    # ax.axhline(y = 0, ls='--', c='b', lw=0.1)
    [ax.axhline(y=i, linestyle='--', lw=0.1) for i in [0,1]]
    # plt.axhline(y = 10)
    # plt.yscale("log")
    ax.set_title("Moment convergence")
    ax.set_xlabel('Iteration')
    ax.legend()
    filename = os.path.join(figures_directory, 'moment_convergence.png')
    plt.tight_layout()
    fig.savefig(filename)



# cwd = os.getcwd()
# metadata = '/drivers/outdir/1606342765/rosenbrock_proper_nP_500_50_1606342765.h5'
# file = cwd + metadata
# f = h5py.File(file)
# d = f['/data']
# plt.style.use('seaborn-dark-palette')
# plt.style.use('seaborn-dark')
# fig, ax = plt.subplots()
# ax.plot(mmd_svn_I_BW, label='SVN MED BW')
# # ax.legend(['SVN MED BW'])
# ax.plot(mmd_svgd, label='SVGD MED BW')
# # ax.legend(['SVGD MED BW'])
# ax.plot(mmd_svn_metric_BW, label='SVN METRIC BW')
# # ax.legend(['SVN METRIC BW'])
# ax.plot(mmd_svgd_BMBW, label='SVGD BMBW')
# ax.plot(mmd_SVN_BMBW, label='SVN BMBW')
# ax.set_title("MMD vs Iteration")
# ax.set_xlabel('Iteration')
# ax.set_ylabel('Log MMD')
# plt.legend()