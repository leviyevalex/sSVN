import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
from plots.plot_helper_functions import set_size, collect_samples
import h5py
import logging.config
import os
from cycler import cycler
from models.HRD import hybrid_rosenbrock
from corner import corner
import palettable
# history_path = ''
log = logging.getLogger(__name__)
root = os.path.dirname(os.path.abspath(__file__))

def make_corner_plots(runs, samples_GT=None, save_path=None):
    def getAxesLabels():
        labels = []
        for d in range(samples_GT.shape[1]):
            labels.append('$x_{%i}$' % (d + 1))
        return labels
    labels = getAxesLabels()
    ####################################################
    # Plot settings
    ####################################################
    plt.style.use(os.path.join(root, 'latex.mplstyle'))
    width = 469.75502
    plt.rcParams['figure.figsize']=set_size(width, fraction=1/2, subplots=(1, 1))
    # fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
    # fig, ax = plt.subplots(10, 10, figsize=set_size(width, fraction=2, subplots=(10, 10)))
    # fig, ax = plt.subplots(10, 10, figsize=(470, 470))
    # p = palettable.colorbrewer.qualitative.Set1_3.mpl_colors
    # mpl.rcParams['axes.prop_cycle'] = cycler(color=p)
    c_list = plt.cm.Set1.colors
    n_truth = samples_GT.shape[0]
    # black_line = mlines.Line2D([], [], color='k', label='Truth')
    # red_line = mlines.Line2D([], [], color='red', label='sSVGD')
    # blue_line = mlines.Line2D([], [], color='blue', label='sSVN')

    fig = corner(samples_GT, color='k', weights=np.ones(n_truth)/n_truth, labels=labels, label_kwargs={"fontsize": 25, 'rotation':0}, rasterized=True)
    for method in runs.keys():
        corner()



    # fig = corner(samples_GT, color='k', labels=labels, label_kwargs={"fontsize": 25, 'rotation':0}, rasterized=True)
    for method in runs:
        if method == 'SVGD':
            window = None
            # window = [13000,15000]
            # window = [129000,130000]
            c = c_list[0]
        elif method == 'SVN':
            # window = [1000,2000]
            window = None
            c = c_list[1]
        samples = collect_samples(runs[method], skipsize=10, window=window)
        # samples = collect_samples(h5_dict[key], skipsize=10)
        n = samples.shape[0]
        print('samples: %i' % n)
        corner(samples, fig=fig, color=c, weights=np.ones(n)/n, label=method)
        # corner(samples, fig=fig, color=c, label=key)

    # plt.legend(handles=[blue_line,red_line, black_line], bbox_to_anchor=(0.5, 10, 0.5, 0.5), fontsize=25)
    plt.legend(handles=[blue_line, black_line], bbox_to_anchor=(0.5, 10, 0.5, 0.5), fontsize=25)

    # ax.set_box_aspect(1)
    if save_path is None:
        fig.savefig('cornerKDE.pdf', bbox_inches='tight')
        # g.fig.savefig('cornerKDE.pdf', bbox_inches='tight', dpi=1200)
    else:
        fig.savefig(os.path.join(save_path, 'cornerKDE.png'), bbox_inches='tight', dpi=1200)
        # g.fig.savefig(os.path.join(save_path, 'cornerKDE.pdf'), bbox_inches='tight', dpi=1200)
    pass
    log.info('INFO: Successfully created corner plot.')

def main():
    from pathlib import Path
    # svn_directory = Path(os.getcwd()).parent
    ###########################################
    # Settings for 5D-HRD (Good Deterministic)
    ###########################################
    # n2 = 2
    # n1 = 3
    # HRD = hybrid_rosenbrock(n2=n2, n1=n1, mu_rosen=1, a=30, b=np.ones((n2, n1-1)) * 20)
    # GT = HRD.newDrawFromLikelihood(100000)
    ###########################################
    # Settings for 10D-HRD
    ###########################################
    n2 = 3
    n1 = 4
    HRD = hybrid_rosenbrock(n2=n2, n1=n1, mu=1, a=30, b=np.ones((n2, n1-1)) * 20)
    GT = HRD.newDrawFromLikelihood(2000000)
    svn_directory = Path(os.getcwd()).parent
    # file_svgd = os.path.join(svn_directory, 'experiment_data', '1-dynamics-comparison', 'svgd10d15k.h5')
    file_svn = os.path.join(svn_directory, 'experiment_data', '1-dynamics-comparison', 'svn10d15k.h5')
    # file_dynesty = os.path.join(svn_directory, )
    # file_SVN_det =
    # file_svgd = os.path.join(svn_directory, 'outdir', '1635285342', 'output_data_new.h5')
    # file_svgd = os.path.join(svn_directory, 'outdir', '1635306858', 'output_data_new.h5')


    # make_corner_plots({'sSVGD': file_svgd, 'sSVN': file_svn}, samples_GT=GT)
    make_corner_plots({'sSVN': file_svn}, samples_GT=GT)
    # make_corner_plots({'sSVGD': file_svgd}, samples_GT=GT)




    #########################################
    # Settings for 2D-Gaussian (Stochastic)
    #########################################
    # file1 = os.path.join(svn_directory, 'outdir', '1631428969_gaussian_30k_svn', 'output_data_new.h5')
    # file2 = os.path.join(svn_directory, 'outdir', '1631219554_gaussian_30k_svgd', 'output_data_new.h5')
    # h5_dict = {'sSVN':file1, 'sSVGD': file2}
    # GT_test = np.random.multivariate_normal(np.ones(2),np.eye(2),37500)
    # Settings for 2D rosenbrock
    #########################################
    # Settings for 2D-HRD
    #########################################
    # n2 = 1
    # n1 = 2
    # HRD = hybrid_rosenbrock(n2=n2, n1=n1, mu_rosen=1, a=1, b=np.ones((n2, n1-1)) * (100 / 20))
    # GT = HRD.newDrawFromLikelihood(100000)
    # file1 = os.path.join(svn_directory, 'outdir', '1631550427_hrd_30k_ssvgd', 'output_data_new.h5')
    # file2 = os.path.join(svn_directory, 'outdir', '1631553689_hrd_30k_ssvn', 'output_data_new.h5')
    # make_corner_plots({'sSVGD': file1, 'sSVN': file2}, samples_GT=GT)
    #########################################
    # Settings for 5D-HRD
    #########################################
    # n2 = 2
    # n1 = 3
    # HRD = hybrid_rosenbrock(n2=n2, n1=n1, mu_rosen=1, a=1, b=np.ones((n2, n1-1)) * (100 / 20))
    # GT = HRD.newDrawFromLikelihood(100000)
    # file1 = os.path.join(svn_directory, 'outdir', '1630944273_sSVGD_5d_HRD', 'output_data_new.h5')
    # file2 = os.path.join(svn_directory, 'outdir', '1630961909_sSVN_5d_HRD', 'output_data_new.h5')
    # make_corner_plots({'sSVGD': file1, 'sSVN': file2}, samples_GT=GT)

if __name__ == '__main__':
    main()