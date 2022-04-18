import os
import matplotlib.pyplot as plt
import logging.config
import palettable
from plots.plot_helper_functions import set_size, extract_moments
import numpy as np
from models.gauss_stats import gauss_stats
from cycler import cycler
import matplotlib as mpl
log = logging.getLogger(__name__)
root = os.path.dirname(os.path.abspath(__file__))
def plot_moment_convergence(h5_dict, GT=None, save_path=None):
    for key1 in h5_dict:
        file = h5_dict[key1]
    ####################################################
    # Plot settings
    ####################################################
    plt.style.use(os.path.join(root, 'latex.mplstyle'))
    width = 469.75502
    fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
    p = palettable.colorbrewer.qualitative.Set1_8.mpl_colors
    mpl.rcParams['axes.prop_cycle'] = cycler(color=p)

    # Extract moments from data
    moment_dict = extract_moments(file)
    if GT is not None:
        mean_GT = np.mean(GT, axis=0)
        cov_GT = np.cov(GT.T)
    for d in range(moment_dict['DoF']):
        # markevery = mean_GT[0].shape[0] / 100
        markevery = 5
        ax.plot(moment_dict['mean_history'][:,d], label=r'$\mu_{%i}$' % (d+1), linewidth=0.6, markevery=markevery, markersize='3', rasterized=True)
        ax.plot(moment_dict['cov_history'][:,d], label=r'$\sigma_{%i %i}$' % (d+1, d+1), linewidth=0.6, markevery=markevery, markersize='3', rasterized=True)
        if GT is not None:
            ax.axhline(mean_GT[d], linestyle='--', c='k', linewidth=0.1)
            ax.axhline(cov_GT[d,d], linestyle='--', c='k', linewidth=0.1)

    ax.set_xlabel(r'Iteration $l$')
    ax.legend()
    if save_path is None:
        fig.savefig('moment_convergence.pdf', bbox_inches='tight', dpi=1200)
    else:
        fig.savefig(os.path.join(save_path, 'moment_convergence.pdf'), bbox_inches='tight', dpi=1200)

    log.info('INFO: Successfully created moment convergence plot')

# with h5py.File(file, 'r') as hf:
#     log.info('INFO: Successfully opened data file')
#     keys = [int(l) for l in list(hf.keys())[:-2]]
#     keys.sort()
#     iters_performed = keys[-1]
#     mean_x = np.zeros(iters_performed)
#     mean_y = np.zeros(iters_performed)
#     cov_x = np.zeros(iters_performed)
#     cov_y = np.zeros(iters_performed)
#
#     for l in range(iters_performed):
#         X = hf['%i' % l]['X'][()]
#         mean = np.mean(X, axis=0)
#         # mean[l] = np.mean(X)
#         cov = np.cov(X.T)
#         mean_x[l] = mean[0]
#         mean_y[l] = mean[1]
#         cov_x[l] = cov[0, 0]
#         cov_y[l] = cov[1, 1]
#     log.
    # fig, ax = plt.subplots(1,1)
    # ax.set_prop_cycle(fancy)
    # ax.set_prop_cycle(fancy)
    # ax.set_prop_cycle('color', palettable.colorbrewer.qualitative.Dark2_8.mpl_colors)
    # ax.set_prop_cycle('color', palettable.scientific.sequential.LaPaz_4.mpl_colors)
    # ax.set_prop_cycle('color', palettable.scientific.sequential.Oslo_5.mpl_colors)
    # ax.set_prop_cycle('color', palettable.scientific.sequential.Oslo_4.mpl_colors)
    # ax.set_prop_cycle('color', palettable.scientific.sequential.Oleron_8.mpl_colors)
    # monochrome = (cycler('color', ['k']) * cycler('linestyle', ['solid', 'loosely dotted', 'loosely dashed', 'loosely dashdotted']))
    # * cycler('linestyle', ['solid', 'dotted'])
    # fancy = (cycler('color', ['#bc80bd' ,'#fb8072', '#b3de69','#fdb462','#fccde5','#8dd3c7','#ffed6f','#bebada','#80b1d3', '#ccebc5', '#d9d9d9']))
    # tex_fonts = {
    #     # Use LaTeX to write all text
    #     "text.usetex": True,
    #     "font.family": "serif",
    #     # Use 10pt font in plots, to match 10pt font in document
    #     "axes.labelsize": 10,
    #     "font.size": 10,
    #     # Make the legend/label fonts a little smaller
    #     "legend.fontsize": 8,
    #     "xtick.labelsize": 8,
    #     "ytick.labelsize": 8
    # }
    # plt.rcParams.update(tex_fonts)
    # plt.rcParams['xtick.direction'] = 'in'
    # plt.rcParams['ytick.direction'] = 'in'
def main():
    gs = gauss_stats(2)
    GT = gs.newDrawFromLikelihood(10000)
    plot_moment_convergence(GT)
if __name__ == '__main__':
    main()