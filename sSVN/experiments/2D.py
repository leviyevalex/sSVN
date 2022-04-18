#%% Import packages
import numpy as np
import h5py
import os
import deepdish as dd
from source.stein_experiments import samplers
from collections import OrderedDict
from models.HRD import hybrid_rosenbrock
from models.rosenbrock_analytic_proper_sample import rosenbrock_analytic as double_banana
from plots.mmd_plots import plotMMD
import matplotlib.pyplot as plt
import palettable
from pathlib import Path
from plots.pretty_contours import collect_history
import seaborn as sns
from matplotlib.lines import Line2D

#%% Find root directory and load double banana ground truth file
root1 = Path(os.getcwd()).parent
root2 = os.getcwd()
if os.path.basename(root1) == 'stein_variational_newton':
    root = root1
else:
    root = root2
ground_truth_double_banana_path = os.path.join(root, 'double_banana_ground_truth.h5')

#%% Load Hybrid Rosenbrock and Double Banana
n2 = 1
n1 = 2
HRD = hybrid_rosenbrock(n2=n2, n1=n1, mu=1, a=0.5, b=np.ones((n2, n1-1)) * 0.5, id='thin-like')
DB = double_banana()
targets_dict = OrderedDict()
targets_dict['HRD'] = HRD
targets_dict['DB'] = DB

#%% Load ground truths
nTruth = 300
ground_truth_HRD = HRD.newDrawFromLikelihood(nTruth)
ground_truth_double_banana = dd.io.load(ground_truth_double_banana_path).samples[np.random.randint(0, 8000, nTruth)] # Load random samples
ground_truth_dict = OrderedDict()
ground_truth_dict['HRD'] = ground_truth_HRD
ground_truth_dict['DB'] = ground_truth_double_banana

#%% Common sampler settings
eps = 0.1
nIter = 200
nParticles = 100
methods = ['sSVN', 'sSVGD', 'SVGD', 'SVN']
# methods = ['sSVGD']

#%% Perform runs
output_paths = OrderedDict()
for target, target_model in targets_dict.items():
    output_paths[target] = OrderedDict()
    for method in methods:
        print('Running %s on %s.' % (method, target))
        s = samplers(model=target_model, nIterations=nIter, nParticles=nParticles)
        output_paths[target]['%s' % method] = s.apply(method=method, eps=eps)['path_to_results']
MMD_evolution_dict = OrderedDict()
for target in targets_dict.keys():
    MMD_evolution_dict[target] = OrderedDict()
    MMD_evolution_dict[target] = plotMMD(output_paths[target], ground_truth_dict[target], maxiter=nIter)

#%% (a) Load data if its already there
output_paths = dd.io.load(os.path.join(root, '2d_experiment_output.h5'))
MMD_evolution_dict = dd.io.load(os.path.join(root, 'MMD_2d.h5'))

#%% (b) If data is not there, store it for reuse
dd.io.save('2d_experiment_output.h5', output_paths)
dd.io.save('MMD_2d.h5', MMD_evolution_dict)

#%% Plot settings
figsize = (3.2621875, 3.2621875)
p = palettable.colorbrewer.qualitative.Set1_8.mpl_colors
fig, ax = plt.subplots(2, 2, figsize=figsize)
plt.style.use(os.path.join(root, 'plots', 'latex.mplstyle'))
color_list = []

#%% Contour images
contour_HRD = os.path.join(root, 'outdir', 'hybrid_rosenbrock_1_2_thin-like_-2.00_2.00.h5')
contour_DB = os.path.join(root, 'outdir', 'rosenbrock_proper_-2.50_2.50.h5')
contours = [contour_HRD, contour_DB]
for i, contour_path in enumerate(contours):
    with h5py.File(contour_path, 'r') as hf:
        X = hf["X"][:]
        Y = hf["Y"][:]
        Z = hf["Z"][:]
    cp = ax[i, 0].contour(X, Y, Z, 9, colors='k', alpha=0.4, linewidths=0.5, zorder=1)

# Settings for HRD
ax[0, 0].set_xlim(-2, 3.5)
ax[0, 0].set_ylim(-3, 11)
# Settings for DB
ax[1, 0].set_xlim(-2, 2)
ax[1, 0].set_ylim(-1, 2)

# Make subplots into squares
for i in range(2):
    for j in range(2):
        ax[i, j].set_box_aspect(1)

#%% Plot MMD
for i, model_name in enumerate(targets_dict.keys()):
    for method in methods:
        ax[i, 1].plot(MMD_evolution_dict[model_name][method], linewidth=0.6, label=method)
# ax[1,1].legend()
#%%
p = sns.color_palette(palette=None)
custom_lines = [Line2D([0], [0], color=p[0], lw=1, marker='s', linestyle='None'),
                Line2D([0], [0], color=p[1], lw=1, marker='s', linestyle='None'),
                Line2D([0], [0], color=p[2], lw=1, marker='s', linestyle='None'),
                Line2D([0], [0], color=p[3], lw=1, marker='s', linestyle='None')]

ax[0,1].legend(custom_lines, ['sSVN', 'sSVGD', 'SVGD', 'SVN'], ncol=1, handletextpad=0.01, labelspacing=0.01, loc='best')
plt.rc('legend',fontsize=8)
fig.show()

#%% Automatically clear MMD part
# counter_MMD = 0
# for model_name in targets_dict.keys():
#     for method in methods:
#         ax[counter_MMD, 1].clear()
#     counter_MMD += 1
# fig.show()

#%% Axes labels
for i in range(2):
    ax[i, 0].set_xlabel(r'$x_1$')
    ax[i, 0].set_ylabel(r'$x_2$', rotation=0)
    ax[i, 1].set_xlabel(r'Iteration $(l)$')
    ax[i, 1].set_ylabel(r'MMD')
fig.show()

#%%
p = sns.color_palette(palette=None)
for particle in [5, 2]:
    for i, model_name in enumerate(targets_dict.keys()):
        color_index = 0
        for method in methods:
            chain = collect_history(output_paths[model_name][method], particle=particle, max_l=nIter)
            ax[i, 0].plot(chain[..., 0], chain[..., 1], '-', linewidth=0.3, alpha=0.8, label=method, zorder=2, color=p[color_index])
            ax[i, 0].scatter(chain[-1, 0], chain[-1, 1], marker='*', alpha=1, zorder=3, color=p[color_index])
            color_index += 1

fig.show()
#%%
fig.savefig('2d-exp.pdf', bbox_inches='tight')
