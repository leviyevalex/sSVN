import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from plots.plot_helper_functions import set_size
from tools.discrepancies import KSD
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
def make_KSD_plots(h5_dict, save_path=None):
    ####################################################
    # Plot settings
    ####################################################
    plt.style.use(os.path.join(root, 'latex.mplstyle'))
    width = 469.75502
    fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1/2, subplots=(1, 1)))
    p = palettable.colorbrewer.qualitative.Set1_3.mpl_colors
    mpl.rcParams['axes.prop_cycle'] = cycler(color=p)
    ax.set_xlabel(r'Iteration $l$')
    ax.set_ylabel(r'KSD')

    num_keys = 0
    for key in h5_dict: # If more than one file
        num_keys += 1
        KSD_array = []
        with h5py.File(h5_dict[key], 'r') as hf:
            try:
                L = hf['metadata']['total_num_iterations'][()]
            except:
                L = hf['metadata']['L'][()]
            # L=500
            for l in range(L):
                gmlpt = hf['%i' % l]['gmlpt'][()]
                X = hf['%i' % l]['X'][()]
                KSD_array.append(np.log10(KSD(X, -1 * gmlpt)))
        ax.plot(KSD_array, label=key, alpha=0.5)
    ax.legend()
    fig.show()
    pass
        # sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, -0.05), ncol=num_keys, title=None, frameon=False)

def main():
    from pathlib import Path
    svn_directory = Path(os.getcwd()).parent

    # file1 = os.path.join(svn_directory, 'outdir', '1632049698_sSVN_perfect_10d', 'output_data_new.h5')
    # file2 = os.path.join(svn_directory, 'outdir', '1632425825', 'output_data_new.h5')
    # make_KSD_plots({'sSVN': file1, 'sSVGD': file2})

    file1 = os.path.join(svn_directory, 'outdir', '1632243563_sSVGD,h=0.01,eps=0.01', 'output_data_new.h5')
    file2 = os.path.join(svn_directory, 'outdir', '1632245560_sSVGD,h=0.01,eps=0.1', 'output_data_new.h5')
    make_KSD_plots({'shitty': file1, 'good': file2})
if __name__ is '__main__':
    main()