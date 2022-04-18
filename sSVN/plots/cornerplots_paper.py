import os
from models.HRD import hybrid_rosenbrock
from corner import corner
from pathlib import Path
import numpy as np
from plots.plot_helper_functions import collect_samples
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

#%% Find root directory and load double banana ground truth file
root1 = Path(os.getcwd()).parent
root2 = os.getcwd()
if os.path.basename(root1) == 'stein_variational_newton':
    root = root1
else:
    root = root2

#%% Load Hybrid Rosenbrock results 5D
sSVGD_path = os.path.join(root, 'outdir', '1642028004sSVGD5d_metric', 'output_data.h5')
sSVN_path = os.path.join(root, 'outdir', '1642033794sSVN5d_metric', 'output_data.h5')

#%% Load Hybrid Rosenbrock results 10D
sSVGD_path = os.path.join(root, 'outdir', '1641943342sSVGD_metric', 'output_data.h5')
sSVN_path = os.path.join(root, 'outdir', '1641932744sSVN_identity_metric', 'output_data.h5')

#%% Collect samples
samples_sSVN = collect_samples(sSVN_path, window=[200, 300])
nSamples = samples_sSVN.shape[0]
samples_sSVGD = collect_samples(sSVGD_path, window=[9900, 10000])

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

#%% Get axes labels
def getAxesLabels(DoF):
    labels = []
    for d in range(DoF):
        labels.append('$x_{%i}$' % (d + 1))
    return labels
labels = getAxesLabels(HRD.DoF)

#%% Set colors and style
color_sSVN = '#e41a1c'
color_truth = 'k'
color_sSVGD = '#377eb8'
plt.style.use(os.path.join(root, 'plots', 'latex.mplstyle'))

#%% Create corner plot (Note: Plot ground truth last for best scaling)
fig = corner(samples_sSVN,
             color=color_sSVN,
             rasterized=True,
             sharey=False,
             plot_contours=True,
             labels=labels,
             label_kwargs={"fontsize": 30, 'labelpad': 10},
             labelpad=1000.,
             smooth=0.5,
             )

corner(ground_truth, fig=fig, plot_contours=False, color=color_truth, rasterized=True, bins=50)

#%% Replace buggy histograms with Seaborne KDE's
axes = np.array(fig.axes).reshape((DoF, DoF))
for i, a in enumerate(axes[(range(DoF), range(DoF))]):
    # Remove buggy histograms
    a.clear()
    # Replace with Seaborn's KDE
    sns.kdeplot(samples_sSVGD[:, i], color=color_sSVGD, shade=True, ax=a)
    sns.kdeplot(samples_sSVN[:, i], color=color_sSVN, shade=True, ax=a)
    sns.kdeplot(ground_truth[:, i], color=color_truth, shade=False, ax=a)
    # Disable default Seaborn labels
    a.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    if i == 0:
        a.tick_params(left=True, labelleft=True, labelrotation=45)
        # a.yaxis.label.set_visible(True)
        a.set_ylabel('$x_{%i}$' % 1, fontsize=30, labelpad=20)
        a.locator_params(axis="y", nbins=6)
    elif i == DoF - 1:
        a.tick_params(bottom=True, labelbottom=True, labelrotation=45)
        # a.xaxis.label.set_visible(True)
        a.set_xlabel('$x_{%i}$' % DoF, fontsize=30, labelpad=0)
        a.locator_params(axis="x", nbins=6)


#%% DEBUG: Settings y-bounds
# for a in axes[np.tril_indices_from(axes)]:
#     a.set_ybound(0)
#     a.set_xbound(-0.3)

#%% Increase tick sizes
for ax in fig.get_axes():
    ax.tick_params(axis='both', labelsize=16)

#%% Assign colors to lines for legend
line_sSVN = mlines.Line2D([], [], color=color_sSVN, label='sSVN')
line_sSVGD = mlines.Line2D([], [], color=color_sSVGD, label='sSVGD')
line_truth = mlines.Line2D([], [], color=color_truth, label='Truth')
if DoF == 10:
    # a.legend(handles=[line_sSVN, line_truth, line_sSVGD], bbox_to_anchor=(0.5, 11.5, 0.5, 0.5), fontsize=35, frameon=False)
    a.legend(handles=[line_sSVN, line_truth, line_sSVGD], bbox_to_anchor=(0.5, 10.8, 0.5, 0.5), fontsize=35, frameon=False)
elif DoF == 5:
    # a.legend(handles=[line_sSVN, line_truth, line_sSVGD], bbox_to_anchor=(0.75, 5.5, 0.5, 0.5), fontsize=35, frameon=False)
    a.legend(handles=[line_sSVN, line_truth, line_sSVGD], bbox_to_anchor=(0.75, 4.8, 0.5, 0.5), fontsize=35, frameon=False)

#%%
fig.show()

#%%
fig.savefig('10d-corner.png')

#%%
fig.savefig('10d-corner.pdf')

#%%
fig.savefig('5d-corner.png')

#%%
fig.savefig('5d-corner.pdf')







