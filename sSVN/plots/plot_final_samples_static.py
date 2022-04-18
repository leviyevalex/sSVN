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
import seaborn as sns
log = logging.getLogger(__name__)
root = os.path.dirname(os.path.abspath(__file__))
def make_final_samples_plot(h5_dict, contour_file_path, samples_GT=None, save_path=None):
    ####################################################
    # Plot settings
    ####################################################
    plt.style.use(os.path.join(root, 'latex.mplstyle'))
    width = 469.75502
    fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
    plt.rcParams['figure.figsize']=set_size(width, fraction=1, subplots=(1, 1))
    p = palettable.colorbrewer.qualitative.Set1_8.mpl_colors
    mpl.rcParams['axes.prop_cycle'] = cycler(color=p)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    # plt.axis('off')
    # ax.set_title('Collected samples')
    num_keys=0

    # Plot ground truth samples
    if samples_GT is not None:
        plt.hist2d(samples_GT[:,0], samples_GT[:,1], bins=100)
        num_keys += 1

    # Plot contours
    with h5py.File(contour_file_path, 'r') as hf:
        X = hf["X"][:]
        Y = hf["Y"][:]
        Z = hf["Z"][:]
    cp = ax.contour(X, Y, Z, 9, colors='r', alpha=0.1)

    # Plot collected samples
    for key in h5_dict:
        num_keys += 1
        samples = collect_samples(h5_dict[key], skipsize=10)
        ax.scatter(samples[:,0], samples[:,1], marker=".", s=1, label=key)

    ax.legend()
    sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, -0.05), ncol=num_keys, title=None, frameon=False)
    if save_path is None:
        fig.savefig('final_samples.png', bbox_inches='tight')
    else:
        fig.savefig(os.path.join(save_path, 'final_samples.png'), bbox_inches='tight')
    log.info('Successfully made final samples plot.')

def main():
    from pathlib import Path
    n2 = 1
    n1 = 2
    HRD = hybrid_rosenbrock(n2=n2, n1=n1, mu_rosen=1, a=0.5, b=np.ones((n2, n1-1)) * 0.5) # nice
    svn_directory = Path(os.getcwd()).parent
    contour_file = os.path.join(svn_directory, 'outdir', 'hybrid_rosenbrock_1_2_-10.00_10.00.h5')
    #########################################
    # Settings for 2D-HRD
    #########################################
    n2 = 1
    n1 = 2
    HRD = hybrid_rosenbrock(n2=n2, n1=n1, mu_rosen=1, a=1, b=np.ones((n2, n1-1)) * (100 / 20))
    GT = HRD.newDrawFromLikelihood(2000000)
    svn_directory = Path(os.getcwd()).parent
    file1 = os.path.join(svn_directory, 'outdir', '1631550427_hrd_30k_ssvgd', 'output_data_new.h5')
    file2 = os.path.join(svn_directory, 'outdir', '1631553689_hrd_30k_ssvn', 'output_data_new.h5')
    contour_file = os.path.join(svn_directory, 'outdir', 'hybrid_rosenbrock_1_2_-10.00_10.00.h5')
    make_final_samples_plot({'sSVGD': file1, 'sSVN': file2}, contour_file_path=contour_file, samples_GT=GT)



    #########################################
    # Settings for 5D-HRD (ORIGINAL)
    #########################################
    # n2 = 2
    # n1 = 3
    # HRD = hybrid_rosenbrock(n2=n2, n1=n1, mu_rosen=1, a=1/20, b=np.ones((n2, n1-1)) * (100 / 20))
    # GT = HRD.newDrawFromLikelihood(2000000)
    # log.info('Samples drawn')
    # from corner import corner
    # fig = corner(GT)
    # plt.show()
    # pass
    # fig.savefig('test_HRD_5D_corner.png')

if __name__ is '__main__':
    main()