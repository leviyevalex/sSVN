import numpy as np
import matplotlib.pyplot as plt
import h5py
import logging.config
import os
from opt_einsum import contract
import scipy
root = os.path.dirname(os.path.abspath(__file__))
log = logging.getLogger(__name__)
from tools.kernels import getIMQ_kernelWithDerivatives_identity, Gaussian_identity
def KSD(X, score):
    kx_IMQ, gkx_IMQ, hesskx_IMQ = getIMQ_kernelWithDerivatives_identity(X, c=1, beta=-0.5, get_hesskx=True)
    KSD2 = (- contract('mnii ->', hesskx_IMQ)
            + contract('mni, ni -> ', gkx_IMQ, score)
            - contract('mni, mi -> ', gkx_IMQ, score)
            + contract('mn, mi, ni ->', kx_IMQ, score, score)) / X.shape[0] ** 2
    return np.sqrt(KSD2)

mmd_kernel = Gaussian_identity(h=1).kx
def MMD(X, Y):
    M = X.shape[0]
    N = Y.shape[0]
    return np.sum(mmd_kernel(X, X)) / M ** 2 \
      -2 * np.sum(mmd_kernel(X, Y)) / (M * N) \
         + np.sum(mmd_kernel(Y, Y)) / N ** 2









    # Kernel Stein divergence (KSD)


    # # (Measuring Sample Quality with Kernels)
    # a1 = np.einsum('mi, ni, mn -> i', score, score, kx_IMQ)
    # b1 = np.einsum('mi, mni -> i', score, gkx_IMQ)
    # c1 = -np.einsum('mni, ni -> i', gkx_IMQ, score)
    # d1 = np.einsum('mn, mni -> i', 4 * beta * (beta - 1) * tmp1 ** (beta - 2), pairwise_displacement ** 2) + np.einsum('mn, i -> i', tmp2, np.ones(DoF))
    # # d1 = -np.einsum('mni -> i', diag_hesskx_IMQ_ensemble)
    # w = np.sqrt((a1 + b1 + c1 - d1) / m ** 2)
    #
    # KSD = np.linalg.norm(w)
    # return KSD


    # Get KSD
    # a1 = np.einsum('mi, ni, mn -> ', gmlpt, gmlpt, kx_IMQ_ensemble)
    # b1 = np.einsum('mi, mni -> ', gmlpt, gkx_IMQ_ensemble)
    # c1 = -np.einsum('mni, ni -> ', gkx_IMQ_ensemble, gmlpt)
    # d1 = -np.einsum('mn ->', tr_hesskx_IMQ_ensemble)
    # KSD = (a1 + b1 + c1 + d1) / m ** 2
    # return KSD
# getPairwiseDisplacement = lambda X: X[:,np.newaxis,:] - X[np.newaxis,:,:] # Way simpler this way
# getPairwiseDistance = lambda X: scipy.spatial.distance.cdist(X, X)
# # Inter-particle calculations
# pairwise_displacement = getPairwiseDisplacement(X)
# pairwise_distances = getPairwiseDistance(X)
#
# # Convenient temporary variables for kernel evaluations
# tmp1 = (c ** 2 + pairwise_distances ** 2)
# tmp2 = 2 * beta * tmp1 ** (beta - 1)
#
# # Get kernel, kernel gradient, and the kernel Hessian trace (inverse multi-quadratic kernel)
# kx_IMQ_ensemble = tmp1 ** beta
# gkx_IMQ_ensemble = np.einsum('mn, mni -> mni', tmp2, pairwise_displacement)
# tr_hesskx_IMQ_ensemble = np.einsum('mn, mn -> mn', 4 * beta * (beta - 1) * tmp1 ** (beta - 2), pairwise_distances ** 2) + DoF * tmp2

# diag_hesskx_IMQ_ensemble = np.einsum('mn, mni -> mni', 4 * beta * (beta - 1) * tmp1 ** (beta - 2), pairwise_displacement ** 2) + np.einsum('mn, i -> mni', tmp2, np.ones(DoF))
# diag_hesskx_IMQ_ensemble = np.einsum('mn, mni -> mni', 4 * beta * (beta - 1) * tmp1 ** (beta - 2), pairwise_displacement ** 2) + np.einsum('mn, i -> mni', tmp2, np.ones(DoF))
# Test kernel gradients
# kx_IMQ_individual = lambda x, y: (c ** 2 + np.linalg.norm(x-y) ** 2) ** beta
# gkx_IMQ_individual = nd.Jacobian(kx_IMQ_individual)
# hesskx_IMQ_individual = nd.Hessian(kx_IMQ_individual)
# for m, n in itertools.product(range(25), range(25)):
#     assert np.allclose(kx_IMQ_individual(X[m], X[n]), kx_IMQ_ensemble[m,n])
#     assert np.allclose(gkx_IMQ_individual(X[m], X[n]), gkx_IMQ_ensemble[m,n])
#     assert np.allclose(np.trace(hesskx_IMQ_individual(X[m], X[n])), tr_hesskx_IMQ_ensemble[m,n], 1e-6)
#     assert np.allclose(hesskx_IMQ_individual(X[m], X[n])[range(DoF), range(DoF)], diag_hesskx_IMQ_ensemble[m,n], 1e-6)
# make_KSD_plots({'sSVN': file1})
# with h5py.File(file, 'r') as hf:
#     iter = 50
#     X = hf['%i' % iter]['X'][()]
#     gmlpt = hf['%i' % iter]['gmlpt']
# Single input
# Define needed functions
# For testing purposes
# hess_kx_IMQ_ensemble = np.einsum('mn, mnj, mni -> mnij', 4 * beta * (beta-1) * tmp1 ** (beta - 2), pairwise_displacement, pairwise_displacement) + np.einsum('mn, ij -> mnij', tmp2, np.eye(DoF))
# testc = np.einsum('mnii -> mn', np.einsum('mn, mnj, mni -> mnij', 4 * beta * (beta-1) * tmp1 ** (beta - 2), pairwise_displacement, pairwise_displacement))
# testd =
# pass
# norm_deltas =
# kx_IMQ_ensemble = lambda X: (c ** 2 + scipy.spatial.distance.cdist(X, X) ** 2) ** beta
# gkx_IMQ = lambda X: 2 * beta *



# Calculate the kernel Stein divergence (uses fact that IMQ is translation invariant)
# k_p = -hesskx_IMQ + contract('mni, ni -> mn', gkx_IMQ, score) - contract('mni, mi -> mn', gkx_IMQ, score) + \
#                                                             + contract('mn, mi, ni -> mn', kx_IMQ, score, score)

def main():
    from pathlib import Path
    svn_directory = Path(os.getcwd()).parent
    file = os.path.join(svn_directory, 'outdir', '1632049698_sSVN_perfect_10d', 'output_data_new.h5')
    with h5py.File(file, 'r') as hf:
        L = hf['metadata']['total_num_iterations'][()]
        KSD_array = []
        for l in range(1000):
            gmlpt = hf['%i' % l]['gmlpt'][()]
            X = hf['%i' % l]['X'][()]
            KSD_array.append(KSD(X, -1 * gmlpt))
        plt.plot(KSD_array)
        plt.show()
        pass

if __name__ is '__main__':
    main()