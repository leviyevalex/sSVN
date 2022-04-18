#%%
from pathlib import Path
import os
import matplotlib.pyplot as plt
import palettable
from plots.plot_helper_functions import set_size, extract_moments
import numpy as np
from models.HRD import hybrid_rosenbrock
from plots.plot_helper_functions import collect_samples
import seaborn as sns

#%% Find root directory and load double banana ground truth file
root1 = Path(os.getcwd()).parent
root2 = os.getcwd()
if os.path.basename(root1) == 'stein_variational_newton':
    root = root1
else:
    root = root2

#%% sSVGD - Load Hybrid Rosenbrock results and extract moments
sSVGD_path = os.path.join(root, 'outdir', '1642028004sSVGD5d_metric', 'output_data.h5') # 5-dimensional
# sSVGD_path = os.path.join(root, 'outdir', '1641943342sSVGD_metric', 'output_data.h5') # 10-dimensional
moment_dict = extract_moments(sSVGD_path)
samples_sSVGD = collect_samples(sSVGD_path, window=[9900, 10000])
nSamples = samples_sSVGD.shape[0]
method = 'sSVGD'

#%% sSVN - Load Hybrid Rosenbrock results and extract moments
sSVN_path = os.path.join(root, 'outdir', '1642033794sSVN5d_metric', 'output_data.h5') # 5-dimensional
# sSVN_path = os.path.join(root, 'outdir', '1641932744sSVN_identity_metric', 'output_data.h5') # 10-dimensional
moment_dict = extract_moments(sSVN_path)
samples_sSVN = collect_samples(sSVN_path, window=[200, 300])
nSamples = samples_sSVN.shape[0]
method = 'sSVN'

#%% Load HRD for 5D problem
DoF = 5
n2 = 2
n1 = 3
HRD = hybrid_rosenbrock(n2=n2, n1=n1, mu=1, a=10, b=np.ones((n2, n1-1)) * 30)

#%% Load HRD for 10D problem
DoF = 10
n2 = 3
n1 = 4
HRD = hybrid_rosenbrock(n2=n2, n1=n1, mu=1, a=30, b=np.ones((n2, n1-1)) * 20)

#%% Get ground truth samples
nSamples_truth = nSamples
ground_truth = HRD.newDrawFromLikelihood(nSamples_truth)

#%% Calculate ground truth mean and covariance
mean_GT = np.mean(ground_truth, axis=0)
cov_GT = np.cov(ground_truth.T)

#%% Plot settings
plt.style.use(os.path.join(root, 'plots', 'latex.mplstyle'))
figsize = (3.2621875, 3.2621875)
fig, ax = plt.subplots(1, 1, figsize=figsize)
plt.rc('legend', fontsize=8.6) # For 5D
# plt.rc('legend', fontsize=6) # For 10D
# fig, ax = plt.subplots(1, 1, figsize=(3, 3))

#%% Plot means and variances
for d in range(moment_dict['DoF']):
    relative_error_mean = (moment_dict['mean_history'][:,d] - mean_GT[d]) / mean_GT[d]
    ax.plot(relative_error_mean, label=r'$\mu_{%i}$' % (d+1), linewidth=0.4, alpha=0.8, rasterized=True, linestyle='dashed')
for d in range(moment_dict['DoF']):
    relative_error_var = (moment_dict['cov_history'][:,d] - cov_GT[d,d]) / cov_GT[d,d]
    ax.plot(relative_error_var, label=r'$\sigma_{%i %i}$' % (d+1, d+1), linewidth=0.6, markersize='3', rasterized=True)

#%% Plot settings
ax.set_ybound(lower=-2, upper=2)
ax.axhline(0, linestyle='--', c='k', linewidth=0.5, alpha=0.8)
ax.set_xlabel(r'Iteration $(l)$')
ax.set_ylabel(r'Relative error (%s)' % method)
ax.set_xbound(lower=0, upper=relative_error_mean.shape[0])
ax.legend()
if mean_GT.shape[0] <= 5:
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.04,1.06), ncol=1, title=None, frameon=False)
elif mean_GT.shape[0] > 5:
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.04,1.06), ncol=2, title=None, frameon=False)
ax.set_aspect(1./ax.get_data_ratio())

#%%
# ax.spines['right'].set_visible(True)
# ax.spines['top'].set_visible(True)
# ax.axis('equal')
# ax.set_aspect('equal')

#%%
fig.savefig('moment-convergencesSVN10d.pdf', bbox_inches='tight', dpi=1200)
#%%
fig.savefig('moment-convergence-ssvn.pdf', bbox_inches='tight', dpi=1200)
#%%
fig.savefig('moment-convergence-ssvgd.pdf', bbox_inches='tight', dpi=1200)
#%%
fig.savefig('moments5dsSVGD.pdf', bbox_inches='tight', dpi=1200)
#%%
fig.savefig('moments5dsSVN.pdf', bbox_inches='tight', dpi=1200)



#%% Save
if save_path is None:
    fig.savefig('moment-convergence-ssvn.pdf', bbox_inches='tight', dpi=1200)
else:
    fig.savefig(os.path.join(save_path, 'moment_convergence%s.pdf' % key1), bbox_inches='tight', dpi=1200)

