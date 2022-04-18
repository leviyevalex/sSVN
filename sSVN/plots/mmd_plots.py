import numpy as np
import matplotlib.pyplot as plt
from plots.plot_helper_functions import set_size
import h5py
import logging.config
import os
from models.HRD import hybrid_rosenbrock
import scipy
from pathlib import Path
from collections import OrderedDict
root = Path(os.getcwd()).parent
current_directory = os.path.dirname(os.path.abspath(__file__))
log = logging.getLogger(__name__)
plt.style.use(os.path.join(current_directory, 'latex.mplstyle'))

# Choose bandwidth for MMD kernels
h = 1

# Function to get Gaussian kernel. Takes two ensembles (of possibly different sample sizes) of form N x D
gaussianKernel = lambda X, Y: np.exp(-scipy.spatial.distance.cdist(X, Y) ** 2 / h)

# Function to evaluate the maximum mean discrepancy (MMD). Takes two ensembles (of possibly different sample sizes) of form N x D
getMMD = lambda X, Y: (1 / X.shape[0] ** 2 * np.sum(gaussianKernel(X, X))
                       - 2 / (X.shape[0] * Y.shape[0]) * np.sum(gaussianKernel(X, Y))
                       + 1 / Y.shape[0] ** 2 * np.sum(gaussianKernel(Y, Y))) * (2 * np.pi) ** (-1 * X.shape[1] / 2)

def plotMMD(runs, ground_truth, maxiter=100, ax=None, save_path=None):
    """
    Create maximum mean discrepancy plots from the paper
    Args:
        runs (dict): The key refers to the method used 'SVN', 'sSVGD', etc... The value is the string path to the results
        of that run
        ground_truth (array): An M x d array of ground truth samples from the target distribution
        maxiter (int): How many iterations of the flow to calculate MMD over.
        axis (ax): Matplotlib ax to plot on
        save_path (path): Where to save figure

    Returns (dict): The key refers to the method used 'SVN', 'sSVGD', etc... The value is an (array) representing the
    MMD evaluated over the particle flow.

    """

    # Plot settings
    if ax is None:
        width = 469.75502
        fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1/2, subplots=(1, 1)))
        ax.set_xlabel(r'Iteration $(l)$')
        ax.set_ylabel(r'MMD')

    # Read data and calculate MMD
    MMD_evolution_dict = OrderedDict()
    for method in runs:
        MMD_evolution_dict[method] = OrderedDict()
        log.info('Calculating MMD: %s' % method)
        with h5py.File(runs[method], 'r') as hf:
            mmd_evolution = np.zeros(maxiter)
            for l in range(maxiter):
                X = hf['%i' % l]['X'][()]
                mmd_evolution[l] = getMMD(X, ground_truth)
            markevery = int(maxiter // 10)
            ax.plot(mmd_evolution, label=method, linewidth=0.6, markevery=markevery)
            MMD_evolution_dict[method] = mmd_evolution

    # Legend
    ax.legend()

    # Fix the boundaries of the figure
    ax.set_xbound(lower=0, upper=maxiter)
    plt.gca().set_ylim(bottom=0)

    # Make the figure square
    ax.set_box_aspect(1)

    return MMD_evolution_dict

def main():
    # 10-dimensional Hybrid-Rosenbrock test case
    n2 = 3
    n1 = 4
    HRD = hybrid_rosenbrock(n2=n2, n1=n1, mu=1, a=30, b=np.ones((n2, n1-1)) * 20)
    file_sSVGD = os.path.join(root, 'outdir', '1641943342sSVGD_metric', 'output_data.h5')
    file_sSVN = os.path.join(root, 'outdir', '1641932744sSVN_identity_metric', 'output_data.h5')
    ground_truth = HRD.newDrawFromLikelihood(500)
    runs = {'sSVN': file_sSVN , 'sSVGD': file_sSVGD}
    plotMMD(runs, ground_truth, maxiter=300)
if __name__ is '__main__':
    main()


