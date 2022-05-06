#%%
import os
import matplotlib.pyplot as plt
import logging.config
import palettable
from plots.plot_helper_functions import set_size, extract_moments
import numpy as np
from models.HRD import hybrid_rosenbrock
from cycler import cycler
import matplotlib as mpl
#%%
log = logging.getLogger(__name__)
root = os.path.dirname(os.path.abspath(__file__))
# root = '/mnt/c/Users/Alex/Projects/stein_variational_newton/'
def plot_moment_convergence(h5_dict, GT=None, save_path=None):
    ####################################################
    # Plot settings
    ####################################################
    plt.style.use(os.path.join(root, 'latex.mplstyle'))
    # plt.style.use(os.path.join(root, 'plots', 'latex.mplstyle'))
    width = 469.75502
    fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1/2, subplots=(1, 1)))
    # p = palettable.colorbrewer.qualitative.Set1_8.mpl_colors
    # mpl.rcParams['axes.prop_cycle'] = cycler(color=p)

    # Extract moments from data
    for key in h5_dict:
        key1 = key
        moment_dict = extract_moments(h5_dict[key])
        if GT is not None:
            mean_GT = np.mean(GT, axis=0)
            cov_GT = np.cov(GT.T)

        for d in range(moment_dict['DoF']):
            # relative_error_mean = np.abs((mean_GT[d] - moment_dict['mean_history'][:,d]) / mean_GT[d])
            # relative_error_mean = (mean_GT[d] - moment_dict['mean_history'][:,d]) / mean_GT[d]
            relative_error_mean = (moment_dict['mean_history'][:,d] - mean_GT[d]) / mean_GT[d]
            # markevery = int(np.floor(relative_error_mean.shape[0] / 10))
            ax.plot(relative_error_mean, label=r'$\mu_{%i}$' % (d+1), linewidth=0.4, alpha=0.8, rasterized=True, linestyle='dashed')
        for d in range(moment_dict['DoF']):
            # relative_error_var = np.abs((cov_GT[d,d] - moment_dict['cov_history'][:,d]) / cov_GT[d,d])
            relative_error_var = (moment_dict['cov_history'][:,d] - cov_GT[d,d]) / cov_GT[d,d]
            ax.plot(relative_error_var, label=r'$\sigma_{%i %i}$' % (d+1, d+1), linewidth=0.6, markersize='3', rasterized=True)





    ax.set_ybound(lower=-2, upper=2)
    ax.axhline(0, linestyle='--', c='k', linewidth=0.5, alpha=0.8)
    ax.set_xlabel(r'Iteration $(l)$')
    ax.set_ylabel(r'Relative error')
    ax.set_xbound(lower=0, upper=relative_error_mean.shape[0])
    # plt.legend(bbox_to_anchor=, loc="upper left")
    ax.legend()
    import seaborn as sns
    if mean_GT.shape[0] <= 5:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.04,1.06), ncol=1, title=None, frameon=False)
    elif mean_GT.shape[0] > 5:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.04,1.06), ncol=2, title=None, frameon=False)
    ax.set_aspect(1./ax.get_data_ratio())
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    # ax.axis('equal')
    # ax.set_aspect('equal')
    if save_path is None:
        fig.savefig('moment-convergence%s.pdf' % key1, bbox_inches='tight', dpi=1200)
    else:
        fig.savefig(os.path.join(save_path, 'moment_convergence%s.pdf' % key1), bbox_inches='tight', dpi=1200)

    log.info('INFO: Successfully created moment convergence plot')
#%%
# n2 = 1
# n1 = 2
# model = hybrid_rosenbrock(n2=n2, n1=n1, mu_rosen=1, a=0.5, b=np.ones((n2, n1-1)) * 0.5, id='thin-like')
# GT = model.newDrawFromLikelihood(50000)
# # gs = gauss_stats(2)
# # GT = gs.newDrawFromLikelihood(10000)
# file_svgd = os.path.join(root, 'experiment_data', '1-dynamics-comparison', 'svgd.h5')
# file_svn = os.path.join(root, 'experiment_data', '1-dynamics-comparison', 'svn.h5')
# plot_moment_convergence(file_svgd, GT)
#%%
def main():
    # n2 = 1
    # n1 = 2
    # model = hybrid_rosenbrock(n2=n2, n1=n1, mu_rosen=1, a=0.5, b=np.ones((n2, n1-1)) * 0.5, id='thin-like')
    # GT = model.newDrawFromLikelihood(50000)

    n2 = 3
    n1 = 4
    HRD = hybrid_rosenbrock(n2=n2, n1=n1, mu_rosen=1, a=30, b=np.ones((n2, n1-1)) * 20)
    GT = HRD.newDrawFromLikelihood(2000000)
    from pathlib import Path
    svn_directory = Path(os.getcwd()).parent
    file_svgd = os.path.join(svn_directory, 'experiment_data', '1-dynamics-comparison', 'svgd10d15k.h5')
    file_svn = os.path.join(svn_directory, 'experiment_data', '1-dynamics-comparison', 'svn10d15k.h5')
    # file_svgd = os.path.join(root, 'experiment_data', '1-dynamics-comparison', 'svgd.h5')
    # file_svn = os.path.join(root, 'experiment_data', '1-dynamics-comparison', 'svn.h5')
    plot_moment_convergence({'sSVGD': file_svgd}, GT)
    # plot_moment_convergence({'sSVN': file_svn}, GT)
if __name__ == '__main__':
    main()