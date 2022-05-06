import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from plots.plot_helper_functions import set_size, collect_samples
import h5py
import logging.config
import os
from cycler import cycler
from models.HRD import hybrid_rosenbrock
import palettable
# history_path = ''
log = logging.getLogger(__name__)
root = os.path.dirname(os.path.abspath(__file__))
def make_corner_plots(h5_dict, samples_GT=None, save_path=None):
    ####################################################
    # Plot settings
    ####################################################
    plt.style.use(os.path.join(root, 'latex.mplstyle'))
    width = 469.75502
    plt.rcParams['figure.figsize']=set_size(width, fraction=1, subplots=(1, 1))
    # p = palettable.colorbrewer.qualitative.Set1_3.mpl_colors
    # mpl.rcParams['axes.prop_cycle'] = cycler(color=p)

    ####################################################
    # Use first file in dict to initialize dataframe
    ####################################################
    first_key = list(h5_dict.keys())[0]
    num_keys = 1
    store1 = collect_samples(h5_dict[first_key], skipsize=10)
    N = store1.shape[0]
    df = pd.DataFrame(store1).assign(dataset='%s' % first_key)

    ###################################################
    # Store data in initialized dataframe
    ###################################################
    for key in h5_dict: # If more than one file
        print(key)
        # log.info('%s' % key)
        if key != first_key:
            num_keys += 1
            store2 = collect_samples(h5_dict[key], skipsize=10)
            assert(N == store2.shape[0])
            df_key = pd.DataFrame(store2).assign(dataset='%s' % key)
            df = pd.concat([df, df_key], ignore_index=True)

    if samples_GT is not None: # If GT is given
        GT_df = pd.DataFrame(samples_GT[0:N]).assign(dataset='Truth')
        df = pd.concat([df, GT_df], ignore_index=True)
        num_keys += 1

    # Plot
    # g = sns.pairplot(df, plot_kws={"s": 1, "rasterized":True}, diag_kws={"shade": True, "common_norm": True, "rasterized":True}, hue='dataset', corner=True, diag_kind='kde')
    # log.info('plotting')
    print('plotting')
    g = sns.pairplot(df, plot_kws={}, diag_kws={"shade": True, "common_norm": True}, hue='dataset', corner=True, diag_kind='kde', kind='hist')
    # g = sns.pairplot(df, diag_kws={"shade": True, "common_norm": True}, hue='dataset', corner=True, kind='kde')

    ###############################################################################################################
    # Label with vector components: https://catherineh.github.io/programming/2016/05/24/seaborn-pairgrid-tips
    ###############################################################################################################
    replacements = {}
    DoF = samples_GT[0].shape[0]
    for d in range(DoF):
        replacements['%i' % d] = r'$x_{%i}$' % (d + 1)
    for i in range(DoF):
        for j in range(DoF):
            try:
                xlabel = g.axes[i][j].get_xlabel()
                ylabel = g.axes[i][j].get_ylabel()
            except:
                xlabel = None
                ylabel = None
            if xlabel in replacements.keys():
                g.axes[i][j].set_xlabel(replacements[xlabel])
            if ylabel in replacements.keys():
                g.axes[i][j].set_ylabel(replacements[ylabel], rotation=0, labelpad=12)

    # Axes settings
    sns.move_legend(g, "lower center", bbox_to_anchor=(.5, -0.05), ncol=num_keys, title=None, frameon=False)
    # g.fig.show()

    if save_path is None:
        g.fig.savefig('cornerKDE.png', bbox_inches='tight', dpi=200)
        # g.fig.savefig('cornerKDE.pdf', bbox_inches='tight', dpi=1200)
    else:
        g.fig.savefig(os.path.join(save_path, 'cornerKDE.png'), bbox_inches='tight', dpi=200)
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
    HRD = hybrid_rosenbrock(n2=n2, n1=n1, mu_rosen=1, a=30, b=np.ones((n2, n1-1)) * 20)
    GT = HRD.newDrawFromLikelihood(100000)
    svn_directory = Path(os.getcwd()).parent
    # file_svgd = os.path.join(svn_directory, 'experiment_data', '1-dynamics-comparison', 'svgd10d15k.h5')
    # file_svn = os.path.join(svn_directory, 'experiment_data', '1-dynamics-comparison', 'svn10d15k.h5')
    file_svgd = os.path.join(svn_directory, 'outdir', '1635285342', 'output_data_new.h5')
    make_corner_plots({'sSVGD': file_svgd}, samples_GT=GT)




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