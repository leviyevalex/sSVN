import os
from models.HRD import hybrid_rosenbrock
from corner import corner
from pathlib import Path
import numpy as np
from plots.plot_helper_functions import set_size, collect_samples
import matplotlib.pyplot as plt
import deepdish as dd
import pandas as pd
import seaborn as sns
from itertools import cycle

#%% # Setup colors for Seaborne
colors = ['#e41a1c', '#377eb8']
# colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']
customPalette = sns.color_palette(colors)
sns.set_palette(customPalette)

#%% Find root directory
root1 = Path(os.getcwd()).parent
root2 = os.getcwd()
if os.path.basename(root1) == 'stein_variational_newton':
    root = root1
else:
    root = root2

#%% Choose case
DoF = 5
# DoF = 10

#%% Load Hybrid Rosenbrock results 5D
if DoF == 5:
    sSVGD_path = os.path.join(root, 'outdir', '1642028004sSVGD5d_metric', 'output_data.h5')
    sSVN_path = os.path.join(root, 'outdir', '1642033794sSVN5d_metric', 'output_data.h5')

#%% Load Hybrid Rosenbrock results 10D
if DoF == 10:
    sSVGD_path = os.path.join(root, 'outdir', '1641943342sSVGD_metric', 'output_data.h5')
    sSVN_path = os.path.join(root, 'outdir', '1641932744sSVN_identity_metric', 'output_data.h5')
    # dynesty_path = os.path.join(root, '10d-dynesty-res.h5')
    # dynesty_path = os.path.join(root, 'dynesty_new_results.h5')

#%% Collect samples
samples_sSVN = collect_samples(sSVN_path, window=[200, 300])
samples_sSVGD = collect_samples(sSVGD_path, window=[9900, 10000])
nSamples = samples_sSVN.shape[0]
# samples_Dynesty = dd.io.load(dynesty_path).samples[0:nSamples]

#%% Load HRD for 5D problem
if DoF == 5:
    n2 = 2
    n1 = 3
    HRD = hybrid_rosenbrock(n2=n2, n1=n1, mu=1, a=10, b=np.ones((n2, n1-1)) * 30)

#%% Load HRD for 10D problem
if DoF == 10:
    n2 = 3
    n1 = 4
    HRD = hybrid_rosenbrock(n2=n2, n1=n1, mu=1, a=30, b=np.ones((n2, n1-1)) * 20)

#%% Get ground truth samples
nSamples_truth = nSamples
ground_truth = HRD.newDrawFromLikelihood(nSamples_truth)
# labels = getAxesLabels(HRD.DoF)

#%% Concatenate all samples
df_sSVGD = pd.DataFrame(samples_sSVGD).assign(Method='sSVGD')
df_sSVN = pd.DataFrame(samples_sSVN).assign(Method='sSVN')
df_truth = pd.DataFrame(ground_truth).assign(Method='Truth')
df = pd.concat([df_sSVGD, df_sSVN, df_truth], ignore_index=True)

#%% Plot style settings
plt.style.use(os.path.join(root, 'plots', 'latex.mplstyle'))

#%% Plot KDE on diagonal of Pairgraid
color_truth = 'k'
def make_kde(*args, **kwargs):
    if kwargs['label'] == 'Truth':
        kwargs['color'] = color_truth
        sns.kdeplot(*args, shade=False, **kwargs, linewidth=0.5, alpha=1)
    elif kwargs['label'] == 'sSVN':
        sns.kdeplot(*args, shade=True, linewidth=0.2, alpha=0.5, **kwargs)
    else:
        sns.kdeplot(*args, shade=True, alpha=0.3, linewidth=0.1, **kwargs)
pg = sns.PairGrid(df, hue='Method', diag_sharey=False, palette=customPalette, corner=True, height=1)
pg.map_diag(make_kde, common_norm='True')
# pg.map_diag(make_kde, common_norm='True')

#%% Label with LaTeX coordinates
replacements = {}
for d in range(DoF):
    replacements['%i' % d] = r'$x_{%i}$' % (d + 1)
for i in range(DoF):
    for j in range(DoF):
        try:
            xlabel = pg.axes[i][j].get_xlabel()
            ylabel = pg.axes[i][j].get_ylabel()
        except:
            xlabel = None
            ylabel = None
        if xlabel in replacements.keys():
            pg.axes[i][j].set_xlabel(replacements[xlabel])
        if ylabel in replacements.keys():
            pg.axes[i][j].set_ylabel(replacements[ylabel], rotation=0, labelpad=14)

#%%
if colors[0] != '#000000':
    colors.insert(0, '#000000')
customPalette = sns.color_palette(colors)

#%%
pg.map_lower(sns.histplot, palette=customPalette)

#%%
pg.fig.savefig(os.path.join(root, 'test_corner.pdf'))

#%%
def make_2dhist(*args, **kwargs):
    if kwargs['label'] == 'Truth':
        plt.hist2d(*args, bins=1000, cmap='Blues')
pg.map_lower(make_2dhist)

#%% # Plot Heatmap and scatter on lower triangular PairGrid
# class heatmap_scatter:
#     def __init(self):
#         self.g = None
#     def make_heatmap_scatter(self, *args, **kwargs):
#         if kwargs['label'] == 'Truth':
#             self.g = sns.jointplot(*args, kind="hex", marginal_kws=dict(bins=100))
#         else:
#             sns.scatterplot(*args, s=5, color=".15", ax=self.g.ax_joint)
# heatmap = heatmap_scatter()
#%%
def make_hist(*args, **kwargs):
    df = pd.DataFrame(args)
    if kwargs['label'] == 'Truth':
        kwargs['color'] = color_truth
        sns.histplot(df, **kwargs)
    else:
        sns.histplot(df, **kwargs)
#%%
# Set custom color palette
if colors[0] != '#000000':
    colors.insert(0, '#000000')
customPalette = sns.color_palette(colors)
pg.map_lower(sns.kdeplot, palette=customPalette, rasterized=True)
# sns.set_palette(customPalette2)
#%%
pg.map_lower(sns.histplot, rasterized=True)
#%%
pg.map_lower(make_hist, rasterized=True)

#%%
# pg.map_lower(heatmap.make_heatmap_scatter)
#%%
pg.fig.savefig(os.path.join(root, 'test_corner.pdf'))
#%%
pg.fig.savefig(os.path.join(root, 'test_corner.png'))


#%% Dynesty related stuff

# df_Dynesty = pd.DataFrame(samples_Dynesty).assign(dataset='Dynesty')
