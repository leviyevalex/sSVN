import matplotlib.pyplot as plt
import h5py
import os
import os
from models.HRD import hybrid_rosenbrock
import numpy as np
from cycler import cycler
import matplotlib as mpl
from plots.plot_helper_functions import set_size, collect_samples
import logging.config
import palettable
# from plots.plot_helper_functions import collect_samples
import seaborn as sns
root = os.path.dirname(os.path.abspath(__file__))
# particle = 30 # This one shows the idea
#%%
#############################################################
# Define plot settings
#############################################################
# plt.style.use(os.path.join(root, 'plots', 'latex.mplstyle'))
# plt.style.use(os.path.join(root, 'plots', 'latex.mplstyle'))
width = 469.75502
fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1/2, subplots=(1, 1)))
# plt.rcParams['figure.figsize']=set_size(width, fraction=1, subplots=(1, 1))
p = palettable.colorbrewer.qualitative.Set1_8.mpl_colors
mpl.rcParams['axes.prop_cycle'] = cycler(color=p)
#############################################################
# Create contour
#############################################################
# contour_file_path = 'hybrid_rosenbrock_1_2_thin-like_-2.00_2.00.h5'
# file_contour = os.path.join('outdir', contour_file_path)
# with h5py.File(file_contour, 'r') as hf:
#     X = hf["X"][:]
#     Y = hf["Y"][:]
#     Z = hf["Z"][:]
# cp = ax.contour(X, Y, Z, 7, colors='k', alpha=0.2, linewidths=0.5, extent=[-0.5,2, -0.5, 2])
# ax.axis('equal')
#############################################################
# Collect a particles history
#############################################################
# file_svgd = os.path.join(root, 'experiment_data', '1-dynamics-comparison', 'svgd.h5')
# file_svn = os.path.join(root, 'experiment_data', '1-dynamics-comparison', 'svn.h5')
def collect_history(file, particle, max_l=None):
    with h5py.File(file, 'r') as hf:
        L = hf['metadata']['L'][()]
        if L > max_l:
            L = max_l
        # nParticles = hf['metadata']['nParticles'][()]
        # print(nParticles)
        DoF = hf['metadata']['DoF'][()]
        history = np.zeros((L, DoF))
        for l in range(L):
            particles_l = hf['%i' % l]['X'][()]
            history[l, :] = particles_l[particle, :]
    return history
#############################################################
# Plot particle trajectory
#############################################################
#%%
particle = np.random.randint(0, 100, 1)
# exploring the mean
# particle = 38
# exploring the tail
# particle = 30
# particle = 18
# particle = 77
# # file = file_svgd
# # file = file_svn
# counter = 0
# for file in (file_svgd, file_svn):
#     with h5py.File(file, 'r') as hf:
#         # L = hf['metadata']['L'][()]
#         if counter == 0:
#             counter += 1
#             print('good')
#             x0 = hf['0']['X'][()][:,0]
#             y0 = hf['0']['X'][()][:,1]
#             ax.scatter(x0, y0, c='k', alpha=0.5, marker='o', s=0.3)
#
#         nParticles = hf['metadata']['nParticles'][()]
#         history = collect_history(file, particle, max_l=500)
#         if file is file_svgd:
#             label='sSVGD'
#         else:
#             label='sSVN'
#         ax.plot(history[..., 0], history[..., 1], '-', linewidth=0.3, alpha=1, label=label)
# ax.set_aspect('auto')
# ax.set_xbound(lower=-1.5, upper=3.2)
# ax.set_ybound(lower=-6, upper=12)
# # ax.axes.xaxis.set_visible(False)
# # ax.axes.yaxis.set_visible(False)
# ax.spines['right'].set_visible(True)
# ax.spines['top'].set_visible(True)
# ax.legend(loc='upper left')
# fig.show()
# fig.savefig('pretty-contour-test.pdf', bbox_inches='tight', dpi=1200)
