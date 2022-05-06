import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from plots.plot_helper_functions import set_size
import h5py
import logging.config
import os
from models.HRD import hybrid_rosenbrock
import palettable
from plots.plot_helper_functions import collect_samples
# from statsmodels.graphics.gofplots import qqplot_2samples
import scipy
from cycler import cycler
import matplotlib as mpl
root = os.path.dirname(os.path.abspath(__file__))
log = logging.getLogger(__name__)
def make_pp_plots(h5_dict, GT, save_path=None):
    ####################################################
    # Plot settings
    ####################################################
    plt.style.use(os.path.join(root, 'latex.mplstyle'))
    width = 469.75502
    fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1/2, subplots=(1, 1)))
    # p = palettable.colorbrewer.qualitative.Set1_8.mpl_colors
    # p2 = palettable.colorbrewer.qualitative.Set2_8.mpl_colors
    # mpl.rcParams['axes.prop_cycle'] = cycler(color=(p + p2))

    first_key = list(h5_dict.keys())[0]
    samples = collect_samples(h5_dict[first_key], skipsize=10)
    print(samples.shape)
    with h5py.File(h5_dict[first_key], 'r') as hf:
        # samples = hf['final_updated_particles']['X'][()]
        DoF = hf['metadata']['DoF'][()]

    # ############################################
    # # Add confidence interval
    # ############################################
    # k = np.arange(0, N + 1)
    # p = k / N
    # confidence=0.95
    # ci_lo, ci_hi = scipy.stats.beta.interval(confidence, k + 1, N - k + 1)
    # ax.axes.fill_betweenx(p, ci_lo, ci_hi)

    # Put diagonal line

    # PP-plot for every dimension (works for different size data sets)
    # Solution adapted from https://www.adrian.idv.hk/2021-07-23-qqplot/
    n = GT.shape[0]
    m = samples.shape[0]

    for d in range(DoF):
        plt.plot(np.linspace(0,1,n), np.interp(np.sort(GT[:,d]), np.sort(samples[:,d]), np.linspace(0,1,m)), alpha=1,label=r"$x_{%d}$" % (d+1), linewidth=1)
    #
    # p = np.linspace(0.1,0.9,9)
    # L = []
    # U = []
    # conf = 0.95
    # for p_ in p:
    #     res_ = scipy.stats.binom.interval(conf, n, p_)
    #     res = [int(item) for item in res_]
    #     L.append(np.sort(GT[:,0])[res[0]])
    #     U.append(np.sort(GT[:,0])[res[1]])
    # plt.plot(L)
    # plt.plot(U)
    # a=1+1

    ###############################################################
    # from statsmodels.distributions.empirical_distribution import ECDF
    # alpha = 0.05
    # F_m_hat = ECDF(samples[:,0])(np.sort(samples[:,0]))
    # p_m_hat = F_m_hat / m
    # z_alpha_over_2 = scipy.stats.norm.ppf(alpha/2)
    #################################################
    # 2.1 A Confidence band for \hat{F_n}
    #################################################
    # delta_m = np.sqrt(1 / (2 * m) * np.log(2 / alpha))
    # L = np.maximum(F_m_hat - delta_m, np.zeros(m))
    # U = np.minimum(F_m_hat + delta_m, np.ones(m))
    # plt.plot(np.linspace(0,1,m), L)
    # plt.plot(np.linspace(0,1,m), U)
    #################################################
    # 2.2.2 Asymptotic (Wald)
    #################################################

    # plt.show()
    # pass
    # # Solution adapted from https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/core/result.py
    # x_values = np.linspace(0, 1, 1001)
    # # N = len(credible_levels)
    # N = m
    # confidence_interval=[0.68, 0.95, 0.997]
    # confidence_interval_alpha=[0.1, 0.2, 0.3] # Controls transparency
    # for ci, alpha in zip(confidence_interval, confidence_interval_alpha):
    #     edge_of_bound = (1. - ci) / 2.
    #     lower = scipy.stats.binom.ppf(1 - edge_of_bound, N, x_values) / N
    #     upper = scipy.stats.binom.ppf(edge_of_bound, N, x_values) / N
    #     # The binomial point percent function doesn't always return 0 @ 0,
    #     # so set those bounds explicitly to be sure
    #     lower[0] = 0
    #     upper[0] = 0
    #     ax.fill_between(x_values, lower, upper, alpha=alpha, color='k')
    # plt.show()



    # for d in range(DoF):
    #     plt.scatter(np.linspace(0,1,N), np.interp(np.sort(GT[0:N,d]), np.sort(samples[0:N,d]), np.linspace(0,1,N)), alpha=1, s=2, label=r"$x_%d$" % (d+1))
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle='--', color='k', linewidth=0.8)

    # Testing
    ax.set_xbound(lower=0, upper=1)
    ax.set_ybound(lower=0, upper=1)
    ax.set_box_aspect(1)
    ax.set_xlabel(r'CDF (Truth)')
    ax.set_ylabel(r'CDF (%s)' % first_key)
    # plt.rcParams["legend.loc"] = 'upper left'
    plt.legend(bbox_to_anchor=(1.04,1.06), loc="upper left")
    # plt.show()
    if save_path is None:
        fig.savefig('pp-plot%s.pdf' % first_key, bbox_inches='tight', dpi=1200)
    else:
        fig.savefig(os.path.join(save_path, 'pp-plot.pdf'), bbox_inches='tight', dpi=1200)
    pass
def main():
    from pathlib import Path
    svn_directory = Path(os.getcwd()).parent
    #########################################
    # Settings for 2D-HRD
    #########################################
    # n2 = 1
    # n1 = 2
    # model = hybrid_rosenbrock(n2=n2, n1=n1, mu_rosen=1, a=0.5, b=np.ones((n2, n1-1)) * 0.5, id='thin-like')
    # GT = model.newDrawFromLikelihood(2000000)

    n2 = 3
    n1 = 4
    HRD = hybrid_rosenbrock(n2=n2, n1=n1, mu_rosen=1, a=30, b=np.ones((n2, n1-1)) * 20)
    GT = HRD.newDrawFromLikelihood(2000000)




    # file_svgd = os.path.join(svn_directory, 'experiment_data', '1-dynamics-comparison', 'svgd.h5')

    # For 2d test
    # file_svgd = os.path.join(svn_directory, 'experiment_data', '1-dynamics-comparison', 'svgd-30k.h5')
    # file_svn = os.path.join(svn_directory, 'experiment_data', '1-dynamics-comparison', 'svn.h5')
    # For 10D test
    file_svgd = os.path.join(svn_directory, 'experiment_data', '1-dynamics-comparison', 'svgd10d15k.h5')
    file_svn = os.path.join(svn_directory, 'experiment_data', '1-dynamics-comparison', 'svn10d15k.h5')


    # n2 = 1
    # n1 = 2
    # HRD = hybrid_rosenbrock(n2=n2, n1=n1, mu_rosen=1, a=1, b=np.ones((n2, n1-1)) * (100 / 20))
    # GT = HRD.newDrawFromLikelihood(2000000)
    # file1 = os.path.join(svn_directory, 'outdir', '1631550427_hrd_30k_ssvgd', 'output_data_new.h5')
    # file2 = os.path.join(svn_directory, 'outdir', '1631553689_hrd_30k_ssvn', 'output_data_new.h5')
    # make_pp_plots({'sSVGD': file_svgd}, GT=GT)
    make_pp_plots({'sSVN': file_svn}, GT=GT)
    #########################################
    # Settings for 5D-HRD
    #########################################
    # n2 = 2
    # n1 = 3
    # HRD = hybrid_rosenbrock(n2=n2, n1=n1, mu_rosen=1, a=1, b=np.ones((n2, n1-1)) * (100 / 20))
    # GT = HRD.newDrawFromLikelihood(100000)
    # file1 = os.path.join(svn_directory, 'outdir', '1630944273_sSVGD_5d_HRD', 'output_data_new.h5')
    # file2 = os.path.join(svn_directory, 'outdir', '1630961909_sSVN_5d_HRD', 'output_data_new.h5')
    # make_pp_plots({'sSVGD': file1, 'sSVN': file2}, GT=GT)
if __name__ == '__main__':
    main()