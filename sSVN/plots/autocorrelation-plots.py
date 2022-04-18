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
import statsmodels.api as sm
import palettable
from plots.plot_helper_functions import collect_samples
# from statsmodels.graphics.gofplots import qqplot_2samples
import scipy
from cycler import cycler
import matplotlib as mpl
root = os.path.dirname(os.path.abspath(__file__))
log = logging.getLogger(__name__)

def get_chain(file, max_l=None):
    with h5py.File(file, 'r') as hf:
        L = hf['metadata']['L'][()]
        if max_l is None:
            max_l = L
        N = hf['metadata']['nParticles'][()]
        D = hf['metadata']['DoF'][()]
        chain = np.zeros((L, N, D))
        if L > max_l:
            L = max_l
        for l in range(L):
            chain[l] = hf['%i' % l]['X'][()]
    return chain

def make_autocorrelation_plots(h5_dict, save_path=None):
    ####################################################
    # Plot settings
    ####################################################
    plt.style.use(os.path.join(root, 'latex.mplstyle'))
    width = 469.75502
    fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
    ax.set_xlabel(r'Lag $(k)$')
    ax.set_ylabel(r'Statistical autocorrelation')
    # ax.set_ylabel(r'MMD')
    num_keys = 0
    particle = np.random.randint(0, 100, 1)
    L_max = 5000
    markevery = int(np.floor(L_max / 1))
    print(markevery)

    def autocorr(x):
        result = np.correlate(x, x, mode='full')
        return result[int(np.floor(result.size/2)):]

    def autocorr1(x,lags):
        '''numpy.corrcoef, partial'''

        corr=[1. if l==0 else np.corrcoef(x[l:],x[:-l])[0][1] for l in lags]
        return np.array(corr)

    def acf(x, length=20):
        length = x.shape[0]
        return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1] for i in range(1, length)])

    def acf_ensemble(chain):
        acf_v = np.apply_along_axis(acf, 0, chain)
        # acf_v = np.apply_along_axis(acf, axis=0)(chain)
        tmp = np.mean(acf_v, axis=0)
        return tmp


    for key in h5_dict: # If more than one file
        lags = np.arange(1, L_max)
        chain = get_chain(h5_dict[key])
        # print(chain[:, particle, 0][:])
        # autocorrelation = autocorr(chain[500:5000, particle, 0][:,0])
        autocorrelation = autocorr1(chain[500:5000, particle, 0][:,0], lags)
        # autocorrelation = acf_ensemble(chain)[0]
        # autocorrelation = acf(chain[500:5000, particle, 0][:,0], )
        ax.plot(autocorrelation, label=key, linewidth=0.6, markevery=markevery, c='k')
        # mark_every = np.arange(1, lag, lag/10)
        # sm.graphics.tsa.plot_acf(chain[:, particle, 0], ax, lags=lags, c='k', markevery=markevery, use_vlines=True, vlines_kwargs={'colors':'k'})
    ax.legend()
    # ax.set_xbound(lower=0, upper=L)
    ax.set_title('')
    plt.gca().set_ylim(bottom=0)
    ax.set_box_aspect(1)
    fig.show()
    fig.savefig('autocorrelation-plot.pdf', bbox_inches='tight', dpi=1200)
    pass

def main():
    from pathlib import Path
    root = Path(os.getcwd()).parent
    file_svgd = os.path.join(root, 'experiment_data', '1-dynamics-comparison', 'svgd.h5')
    file_svn = os.path.join(root, 'experiment_data', '1-dynamics-comparison', 'svn.h5')
    # n2 = 1
    # n1 = 2
    # model = hybrid_rosenbrock(n2=n2, n1=n1, mu_rosen=1, a=0.5, b=np.ones((n2, n1-1)) * 0.5, id='thin-like')
    # GT = model.newDrawFromLikelihood(500)
    # 'sSVGD': file_svgd,
    make_autocorrelation_plots({'sSVN': file_svn})
    # make_autocorrelation_plots({'sSVGD': file_svgd})
if __name__ is '__main__':
    main()


