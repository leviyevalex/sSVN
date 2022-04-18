# Code from deterministic 1/5/2020

#########################################
# Complicated terms from sSVN
#########################################

#################################################################################
# Stochastic SVN Methods
#################################################################################
def getSVN_Deterministic_correction(self, kx, gkx, Hesskx, Hmlpt, grad_Hmlpt, A_nndd): # For full Hessian
    R = contract('oije, om, on -> omnije', grad_Hmlpt, kx, kx)
    B = contract('oij, ome, on -> onmije', Hmlpt, gkx, kx)
    C = contract('onie, omj -> omnjie', Hesskx, gkx)

    tmp1 = B + C
    tmp2 = tmp1 + R

    bracket =   contract('onmije, mpif, noje -> pf', tmp2, A_nndd, A_nndd) \
                + contract('omnije, mpif, noje -> pf', B, A_nndd, A_nndd) \
                + contract('omnjie, mpif, noje -> pf', C, A_nndd, A_nndd) \
                - contract('onmije, mpif, nmje -> pf', tmp1, A_nndd, A_nndd) \
                - contract('omnije, mpif, nnje -> pf', B, A_nndd, A_nndd) \
                - contract('omnjie, mpif, nnje -> pf', C, A_nndd, A_nndd) \
 \
    GK = np.einsum('mne, nmie -> mi', gkx, A_nndd) - np.einsum('mne, nnie -> mi', gkx, A_nndd)

    return GK - bracket

def getSVN_BD_Deterministic_correction(self, kx, gkx, Hesskx, Hmlpt, grad_Hmlpt, A_BD_nndd): # For block diagonal hessian
    if self.iter_ == 0:
        self.omije = (self.nParticles, self.nParticles, self.DoF, self.DoF, self.DoF)
        self.contract_ijemo = oe.contract_expression('omije, mpif, moje -> pf', self.omije, A_BD_nndd.shape, A_BD_nndd.shape)
        self.contract_ijemm = oe.contract_expression('omije, mpif, mmje -> pf', self.omije, A_BD_nndd.shape, A_BD_nndd.shape)
        self.contract_jiemo = oe.contract_expression('omjie, mpif, moje -> pf', self.omije, A_BD_nndd.shape, A_BD_nndd.shape)
        self.contract_jiemm = oe.contract_expression('omjie, mpif, mmje -> pf', self.omije, A_BD_nndd.shape, A_BD_nndd.shape)
        self.contract_R = oe.contract_expression('oije, om -> omije', grad_Hmlpt.shape, kx.shape)
        self.contract_B = oe.contract_expression('oij, ome, om -> omije', Hmlpt.shape, gkx.shape, kx.shape)
        self.contract_C = oe.contract_expression('omie, omj -> omjie', Hesskx.shape, gkx.shape)
        self.contract_GK_a = oe.contract_expression('mne, nmie -> mi', gkx.shape, A_BD_nndd.shape)
        self.contract_GK_b = oe.contract_expression('mne, nnie -> mi', gkx.shape, A_BD_nndd.shape)

    R = self.contract_R(grad_Hmlpt, kx ** 2)
    B = self.contract_B(Hmlpt, gkx, kx)
    C = self.contract_C(Hesskx, gkx)

    tmp1 = 2 * B + C
    tmp2 = R + tmp1

    bracket_check = self.contract_ijemo(tmp2, A_BD_nndd, A_BD_nndd) \
                    - self.contract_ijemm(tmp1, A_BD_nndd, A_BD_nndd) \
                    + self.contract_jiemo(C, A_BD_nndd, A_BD_nndd) \
                    - self.contract_jiemm(C, A_BD_nndd, A_BD_nndd)

    # GK = np.einsum('mne, nmie -> mi', gkx, A_BD_nndd) - np.einsum('mne, nnie -> mi', gkx, A_BD_nndd)
    GK = self.contract_GK_a(gkx, A_BD_nndd) - self.contract_GK_b(gkx, A_BD_nndd)

    return GK - bracket_check

def getSVN_direction(self, kx, gkx, gmlpt, A):
    kbar = np.mean(gkx, axis=0)
    if self.iter_ == 0:
        self.contract_directionSVN_a = oe.contract_expression('nmji, no, oj -> mi', A.shape, kx.shape, gmlpt.shape)
        self.contract_directionSVN_b = oe.contract_expression('nmji, nj -> mi', A.shape, kbar.shape)
    return self.contract_directionSVN_a(A, kx, -1 * gmlpt) \
           + self.nParticles * self.contract_directionSVN_b(A, kbar)

# def getSVN_BD_Stochastic_correction(self, kx, L_ndd, B=None):
#     if B is None:
#         B = np.random.normal(0, 1, (self.nParticles, self.DoF))
#     return np.einsum('mn, nie, ne -> mi', kx, L_ndd, B) * np.sqrt(2 / self.nParticles)

# def getA_BD(self, HBD_inv, kx):
#     return np.einsum('mij, mn -> mnij', HBD_inv, kx) / self.nParticles

    # def getKernel_linear_RFG(self, X, M, h):
    #     alpha = 1 / (self.DoF + 1)
    #     beta = 1 / (self.nParticles - self.DoF - 1)
    #     kx_a, gkx1_a, gkx2_a, hesskx12_a = get_kernel_linear(X=X, get_kx=True, get_gkx1=True, get_gkx2=True, get_hesskx12=True) # IMQ, linear
    #     kx_b, gkx1_b, gkx2_b, hesskx12_b = get_kernel_RFG(X=X, h=h, M=M, l=(self.nParticles - self.DoF - 1), get_kx=True, get_gkx1=True, get_gkx2=True, get_hesskx12=True)
    #
    #     kx = kx_a * alpha + kx_b * beta
    #     gkx1 = gkx1_a * alpha + gkx1_b * beta
    #     return kx, gkx1

    def get_A_BD_stable(self, LHBD, kx):
        tmp1 = np.repeat(np.eye(self.DoF)[np.newaxis, ..., np.newaxis], self.nParticles, axis=0)
        tmp2 = np.squeeze(tf.linalg.cholesky_solve(LHBD[:, np.newaxis], tmp1))
        return contract('no, nje -> noje', kx, tmp2) / self.nParticles

    # def getSVN_BD_Stochastic_correction_stable(self, LHBD, kx, B=None):
    #     if B is None:
    #         B = np.random.normal(0, 1, (self.nParticles, self.DoF))
    #     LHBDT = contract('mij -> mji', LHBD)
    #     tmp1 = np.squeeze(tf.linalg.triangular_solve(LHBDT, B[..., np.newaxis]))
    #     return np.sqrt(2 / self.nParticles) * contract('mn, ni -> mi', kx, tmp1)


# def get_v_stc_new(self, kx, UH):
#     # Get low rank approximation of kernel
#     kx_eig = scipy.linalg.eigh(kx)
#     r = self.nParticles / 10
#     kx_low_rank = kx_eig[1][:, -r:] @ np.diag(kx_eig[0][-r:]) @ kx_eig[1][:, -r:].T
#     # Sample from N(0, H^{-1}
#     B = np.random.normal(0, 1, self.dim)
#     tmp1 = scipy.linalg.solve_triangular(UH, B, lower=False).reshape(self.nParticles, self.DoF)
#
# def get_v_curvature(self, kx, gkx, Hmlpt):
#     R = contract('oije, om, on -> omnije', grad_Hmlpt, kx, kx)
#     B = contract('oij, ome, on -> onmije', Hmlpt, gkx, kx)
#     C = contract('onie, omj -> omnjie', Hesskx, gkx)

# def getA_nndd(self, kx, UH):
#     K = self.reshapeNNDDtoNDND(contract('mn, ij -> mnij', kx / self.nParticles, np.eye(self.DoF)))
#     A_nndd = self.reshapeNDNDtoNNDD(tf.linalg.cholesky_solve(UH.T, K).numpy())
#     return A_nndd
#
# def get_v_drift(self, kx, gkx, UH, A_nndd):
#     # For shift invariant kernels
#     K = self.reshapeNNDDtoNDND(contract('mn, ij -> mnij', kx / self.nParticles, np.eye(self.DoF)))
#     A_nndd = self.reshapeNDNDtoNNDD(tf.linalg.cholesky_solve(UH.T, K).numpy())
#     return contract('mne, nmie -> mi', gkx, A_nndd) - contract('mne, nnie -> mi', gkx, A_nndd)


# def H_bar_AD(self, GN_Hmlpt, kernel, gradKernel): # ND x ND tensor
#     dim = self.DoF * self.nParticles
#     if self.iter_ == 0:
#         self.contract_H_bar_1 = oe.contract_expression("xy, xz, xbd -> yzbd", kernel.shape, kernel.shape, GN_Hmlpt.shape)
#         # self.contract_H_bar_2 = oe.contract_expression("xyb, xzd -> yzbd", gradKernel.shape, gradKernel.shape) # OLD HESSIAN
#         self.contract_H_bar_2 = oe.contract_expression("xyb, xzd -> zybd", gradKernel.shape, gradKernel.shape) # NEW HESSIAN
#     return ((self.contract_H_bar_1(kernel, kernel, GN_Hmlpt) + self.contract_H_bar_2(gradKernel, gradKernel)) / self.nParticles).swapaxes(1, 2).reshape(dim, dim)


# def H_bar_BD(self, GN_Hmlpt, kernel, gradKernel):
#     return (np.einsum('xij, xz -> zij', GN_Hmlpt, kernel ** 2) + np.einsum('xzi, xzj -> zij', gradKernel, gradKernel)) / self.nParticles

def H_bar(self, GN_Hmlpt, kernel, gradKernel): # ND x ND tensor
    dim = self.DoF * self.nParticles
    if self.iter_ == 0:
        self.contract_H_bar_1 = oe.contract_expression("xy, xz, xbd -> yzbd", kernel.shape, kernel.shape, GN_Hmlpt.shape)
        # self.contract_H_bar_2 = oe.contract_expression("xyb, xzd -> yzbd", gradKernel.shape, gradKernel.shape) # OLD HESSIAN
        self.contract_H_bar_2 = oe.contract_expression("xyb, xzd -> zybd", gradKernel.shape, gradKernel.shape) # NEW HESSIAN
    return ((self.contract_H_bar_1(kernel, kernel, GN_Hmlpt) + self.contract_H_bar_2(gradKernel, gradKernel)) / self.nParticles).swapaxes(1, 2).reshape(dim, dim)




######################################
# LINESEARCH CODE
######################################
    # def linesearch_armijo(self, X, v, gv, gmlpt):
    #     # Set backtracking hyper-parameters (beta=1e-3 worked okay)
    #     beta = 1e-3
    #     if method == 'SVN':
    #         eps = 1
    #     else:
    #         eps = 0.1
    #
    #     # Necessary condition for non-singular map T (Determinant does not switch signs)
    #     while True:
    #         dets = np.linalg.det(np.eye(self.DoF)[np.newaxis] + eps * gv)
    #         if np.allclose(np.sign(dets[0]), np.sign(dets)):
    #             break
    #         log.info('LINESEARCH: Backtrack MAP SINGULAR')
    #         eps /= 2
    #
    #     # Calculate term to evaluate first Wolfe Condition (Armijo)
    #     mlnpt = self.getMinusLogPosterior_ensemble_new(X)
    #     delta_phi = lambda eps: np.mean(mlnpt + np.log(np.abs(np.linalg.det(np.eye(self.DoF)[np.newaxis] + eps * gv)))
    #                                           - self.getMinusLogPosterior_ensemble_new(X + eps * v))
    #     directional_derivative = (oe.contract('mi, mi -> ', gmlpt, v) - oe.contract('mdd -> ', gv)) / self.nParticles
    #     log.info('LINESEARCH: Stein discrepancy = %f' % (-1 * directional_derivative))
    #     # trigger = 0 # For debugging
    #     while delta_phi(eps) < -1 * beta * eps * directional_derivative:
    #         # trigger += 1 # For debugging
    #         log.info('LINESEARCH: Backtracking ARMIJO FALSE')
    #         eps /= 2
    #         # if trigger <=5:
    #         #     log.info('LINESEARCH: BREAK')
    #             # break
    #     return eps
    #
    # def getJacobianMapSVN(self, alphas, gkx2):
    #     # -1 here because gradient is taken w.r.t second argument for this expression!
    #     if self.iter_ == 0:
    #         self.contract_terms_gradw = oe.contract_expression('nd, nyb -> ydb', (alphas.shape), (gkx2.shape))
    #     # return -1 * np.einsum('nd, nyb -> ydb', alphas, gradKernel)
    #     return self.contract_terms_gradw(alphas, gkx2)
    #
    # def getJacobianMapSVN_new(self, alphas, gkx1):
    #     return contract('mnj, ni -> mij', gkx1, alphas)



    def getJacobianMapSVGD(self, gkx2, gmlpt, hesskx12):
        if self.iter_ == 0:
            self.contract_gkx_gmlpt = oe.contract_expression('mnd, mb -> ndb', gkx2.shape, gmlpt.shape)
        return self.contract_gkx_gmlpt(gkx2, -gmlpt) / self.nParticles + np.mean(hesskx12, axis=0)


#########################################
# other unused methods
#########################################
    def H_bar_nndd(self, GN_Hmlpt, kernel, gradKernel): # N x N x D x D tensor
        if self.iter_ == 0:
            self.contract_H_bar_1 = oe.contract_expression("xy, xz, xbd -> yzbd", kernel.shape, kernel.shape, GN_Hmlpt.shape)
            self.contract_H_bar_2 = oe.contract_expression("xyb, xzd -> zybd", gradKernel.shape, gradKernel.shape) # NEW HESSIAN
        return ((self.contract_H_bar_1(kernel, kernel, GN_Hmlpt) + self.contract_H_bar_2(gradKernel, gradKernel)) / self.nParticles)




##################################
# Imports
##################################
# import scipy
# import numdifftools as nd
# from numpy import linalg as la
# from tools.kernels import getPairwiseDisplacement
# from tools.kernels import get_linear_metric as get_kernel
# FOR RFG + LINEAR ///
# from tools.kernels import get_linear_metric as get_kernel_linear
# from tools.kernels import get_randomRBF_metric as get_kernel_RFG
#///
# from tools.kernels import get_pointwise_preconditioned_RBF as get_ppkernel
# from tools.kernels import get_pointwise_preconditioned_RBF_v2 as get_ppkernel
# from tools.kernels import get_IMQ_metric as get_kernel_IMQ
# from opt_einsum import contract
# import scipy.sparse as sparse
# import sys
# from tools.kernels import getGaussianKernelWithDerivatives_identity
# from tools.kernels import getGaussianKernelWithDerivatives_metric
# from tools.kernels import getIMQ_kernelWithDerivatives as getKernelDerivatives
# from tools.kernels import getIMQ_metricWithDerivatives as getKernelWithDerivatives
# from tools.kernels import getLinear_metricWithDerivatives as getKernelWithDerivatives
# from tools.discrepancies import MMD
# import itertools
# import traceback
# import tensorflow as tf
# import tensorflow_probability as tfp
# import deepdish as dd




# M = np.eye(self.DoF)
# h = self.getBandwidth_new(X, h, M)
# h = self.bandwidth_MED(X)

# switch1 = 50
# if iter_ > switch1:
#     kx, gkx1, gkx2, hesskx12 = get_kernel_RFG(X=X, h=h, M=M, l=self.nParticles, get_kx=True, get_gkx1=True, get_gkx2=True, get_hesskx12=True)
#     eps=1
# if iter_ < switch1:
#     kx, gkx1 = getKernelWithDerivatives(X, M=M, h=h)

# kx, gkx1, gkx2, hesskx12 = get_kernel_linear(X=X, M=M, get_kx=True, get_gkx1=True, get_gkx2=True, get_hesskx12=True) # IMQ, linear
# kx, gkx1, gkx2, hesskx12 = get_kernel_linear(X=X, get_kx=True, get_gkx1=True, get_gkx2=True, get_hesskx12=True) # no metric works beautifully
# IMQ
# kx, gkx1, gkx2, hesskx12 = get_kernel_IMQ(X=X, M=M, h=h, beta=-0.5, get_kx=True, get_gkx1=True, get_gkx2=True, get_hesskx12=True) # IMQ, linear
# POINTWISE PRECONDITIONED
# kx, gkx1 = get_ppkernel(X=X, h=h, Ml=GNHmlpt, get_kx=True, get_gkx1=True, get_gkx2=True, get_hesskx12=True)

# kx, gkx1 = self.getKernel_linear_RFG(X, M, h)

# np.min(scipy.linalg.eigvalsh(H_bar)
# print('minimum eigenvalue', np.min(scipy.linalg.eigvalsh(H_bar)))
# j_vSVN = self.getJacobianMapSVN(alphas, gkx2)
# j_vSVN = self.getJacobianMapSVN_new(alphas, gkx1)
# eps = self.linesearch_armijo(X, v_svn, j_vSVN, gmlpt)
# self.iter_ = 0 # So that methods may be used offline without use of constructMap.
# self.optimizeMethod = optimizeMethod

######################################
# SVGD
######################################
# j_vSVGD = self.getJacobianMapSVGD(gkx2, gmlpt, hesskx12)
# eps = self.linesearch_armijo(X, v_svgd, j_vSVGD, gmlpt)

#######################################
# BDSVN
#######################################

# v_svn = contract('mn, ni -> mi', kx, alphas)
### v0 - Original "block-diagonal" approximation
# v_svn = np.squeeze(tf.linalg.solve(HBD, v_svgd[..., np.newaxis]))
### v1 - Block-diagonal approximation
# alphas = np.squeeze(tf.linalg.solve(HBD, v_svgd[..., np.newaxis]))
# v_svn = contract('xd, xn -> nd', alphas, kx)
### v2 "Block-diagonal" approxmation with Cholesky solve
### v3 Block-diagonal approxmation with Cholesky solve
# alphas = np.squeeze(tf.linalg.cholesky_solve(LHBD, v_svgd[..., np.newaxis]))
# v_svn = contract('xd, xn -> nd', alphas, kx)

### v3 Block-diagonal approximation with Cholesky solve
# alphas = np.squeeze(tf.linalg.cholesky_solve(LHBD, v_svgd[..., np.newaxis]))
# v_svn = np.squeeze(tf.linalg.solve(HBD, v_svgd[..., np.newaxis]))
# v_svn = np.zeros((self.nParticles, self.DoF))
# v_svn = np.squeeze(tf.linalg.triangular_solve(LHBD, v_svgd[..., np.newaxis]))
# v_svn = tf.linalg.triangular_solve(LHBD, v_svgd, lower=True)
# for n in range(self.nParticles):
#     L = scipy.linalg.cholesky(HBD[n], lower=True)
#     v_svn[n] = scipy.linalg.cho_solve((L, True), v_svgd[n])
#     assert np.allclose(HBD[n] @ v_svn[n], v_svgd[n], atol=1e-6)
#     pass
# = scipy.linalg.solve_triangular(LHBD[n], v_svgd[n], lower=True)

###################################
# sSVGD
###################################

# kx, gkx = getKernelWithDerivatives(X_new, h=h)
# eig_sing = 1e-13
# eigvals_kx1 = np.linalg.eigvalsh(kx)
# kx1_num_singular = ((-1 * eig_sing < eigvals_kx1) & (eigvals_kx1 < eig_sing)).sum()
# log.info('ALGORITHM: Minimum eigenvalue of kx: %.2e' % min_eigval_kx1)
# log.info('ALGORITHM: Number of near zero eigenvalues of (kx) in range (%.1e < x < %1e): %.2e' % (-eig_sing, eig_sing, min_eigval_kx1))
log.info('ALGORITHM: Rank of K: %f' % (np.linalg.matrix_rank(kx) * self.DoF))
log.info('ALGORITHM: Rank of K_prime: %f' % (np.linalg.matrix_rank(kx) * self.DoF))
log.info('ALGORITHM: Rank of Cholesky of diffusion matrix: %f' % (np.linalg.matrix_rank(L_kx) * self.DoF))
# eigvals_Lkx = np.min(np.linalg.eigvalsh(L_kx))
# near_singular = np.where(np.logical_and(eigvals_Lkx>=-1e-13, eigvals_Lkx<=1e-13))
# log.info('ALGORITHM: Minimum eigenvalue of L_K: %.2e' % )

####################################
# sSVN
####################################
# alpha, L_kx = self.getMinimumPerturbationCholesky(kx)
# if alpha != 0:
#     log.info('NOTE: KX NEGATIVE DEFINITE and updated')
#     kx += alpha * np.eye(self.nParticles)

###########################################################
# stochastic apply method
###########################################################
def stochastic(self, method='SVGD', eps=0.01, h=0.1):
    """
    Applies sSVN or sSVGD to a set of particles and stores information in an h5 file.
    Args:
        method (str): Choose 'SVGD' for sSVGD, 'SVN' for sSVN
        eps (float): Step-size used in Euler-Maruyama discretization of SDE

    Returns:
        results_dict (dict): outdir_path = output directory path, path_to_results = path to h5 file storing results.

    """

    if self.profile == True:
        profiler = Profiler()
        profiler.start()
    try:
        # np.random.seed(int(time()))
        np.random.seed(1)
        X_new = self.model.newDrawFromPrior(self.nParticles)
        with trange(self.nIterations) as ITER:
            for iter_ in ITER:
                gmlpt_new = self.getGradientMinusLogPosterior_ensemble_new(X_new)
                ####################################################
                # Algorithm update
                ####################################################
                if method == 'sSVGD':
                    Hmlpt_new = self.getGNHessianMinusLogPosterior_ensemble_new(X_new)
                    kx, gkx = getKernelWithDerivatives(X_new, h=h, M=np.mean(Hmlpt_new, axis=0))
                    alpha, L_kx = self.getMinimumPerturbationCholesky(kx)
                    if alpha != 0:
                        kx += alpha * np.eye(self.nParticles)
                    v_svgd = self.getSVGD_direction(kx, gkx, gmlpt_new)
                    v_stc = self.getSVGD_Stochastic_correction(L_kx)
                    update = v_svgd * eps + v_stc * np.sqrt(eps)
                elif method == 'sSVNpert':
                    # kx, gkx = getKernelWithDerivatives(X_new, h=h)
                    Hmlpt_new = self.getGNHessianMinusLogPosterior_ensemble_new(X_new)
                    kx, gkx = getKernelWithDerivatives(X_new, h=h, M=np.mean(Hmlpt_new, axis=0))
                    v_svgd = self.getSVGD_direction(kx, gkx, gmlpt_new)
                    H = self.H_bar(Hmlpt_new, kx, gkx)
                    alpha_H, Chol_H = self.getMinimumPerturbationCholesky(H)
                    if alpha_H != 0:
                        H += np.eye(self.dim) * alpha_H
                        # H[range(self.dim), range(self.dim)] += alpha_H
                    log.info('H perturbation:  %.2e' % alpha_H)
                    alphas = tf.linalg.cholesky_solve(Chol_H, v_svgd.flatten()[..., np.newaxis]).numpy().reshape(self.nParticles, self.DoF)
                    v_svn = contract('xd, xn -> nd', alphas, kx)
                    # update = alphas * eps
                    update = v_svn * eps
                elif method == 'sSVNBD':
                    # kx, gkx = getKernelWithDerivatives(X_new, h=h)
                    Hmlpt_new = self.getGNHessianMinusLogPosterior_ensemble_new(X_new)
                    kx, gkx = getKernelWithDerivatives(X_new, h=h, M=np.mean(Hmlpt_new, axis=0))
                    v_svgd = self.getSVGD_direction(kx, gkx, gmlpt_new)
                    HBD = self.h_ij_BD(Hmlpt_new, kx, gkx)
                    LHBD = tf.linalg.cholesky(HBD)
                    if iter_ < 50:
                        HBDop = tf.linalg.LinearOperatorFullMatrix(HBD, is_self_adjoint=True, is_positive_definite=True)
                        alphas = tf.linalg.experimental.conjugate_gradient(HBDop, tf.constant(v_svgd), max_iter=10).x.numpy()
                    else:
                        alphas = np.squeeze(tf.linalg.cholesky_solve(LHBD, v_svgd[..., np.newaxis])) # agreed
                    v_svn = contract('mn, ni -> mi', kx, alphas)
                    ######################### Getting the noise
                    UHBD = contract('mij -> mji', LHBD)
                    B = np.random.normal(0, 1, (self.nParticles, self.DoF))
                    tmp1 = np.zeros((self.nParticles, self.DoF))
                    for n in range(self.nParticles):
                        tmp1[n] = scipy.linalg.solve_triangular(UHBD[n], B[n])
                    v_stc = np.sqrt(2 / self.nParticles) * contract('mn, ni -> mi', kx, tmp1)
                    # update = alphas * eps
                    # update = v_svn * eps
                    update = v_svn * eps + v_stc * np.sqrt(eps)
                    # self.check_rank_GN(GN_Hmlpt) # (Debug)

                    # eig_sol = tf.linalg.eigh(Hmlpt_tmp)
                    # eigvals = eig_sol[0].numpy()
                    # eigvecs = eig_sol[1].numpy()

                    # eigvals[np.argwhere(eigvals < 0)] = 0
                    # eigvals_new = np.abs(eigvals)

                    # Hmlpt = contract('mad, md, mdb -> mab', eigvecs, eigvals, contract('mij -> mji', eigvecs))
                    # Hmlpt = contract('mad, md, mdb -> mab', eigvecs, eigvals_new, contract('mij -> mji', eigvecs))
                    # M=np.mean(Hmlpt, axis=0)
                    # kx, gkx = get_kernel_RFG(X=X_new, h=h, M=M, l=self.nParticles, get_kx=True, get_gkx1=True)
                elif method == 'sSVN':
                    # h = 2 * self.DoF
                    h = self.DoF
                    Hmlpt = self.getGNHessianMinusLogPosterior_ensemble_new(X_new)
                    # kx, gkx = getKernelWithDerivatives(X_new, h=h, M=np.mean(Hmlpt, axis=0))
                    kx, gkx = getKernelWithDerivatives(X_new, h=h)
                    # alpha, L_kx = self.getMinimumPerturbationCholesky(kx)
                    # if alpha != 0:
                    #     log.info('NOTE: KX NEGATIVE DEFINITE and updated')
                    #     kx += alpha * np.eye(self.nParticles)
                    if self.DoF == 10:
                        if iter_ < 10:
                            tau = 1
                        elif iter_ < 50: # 20 for easier 10D case
                            tau = 0.5
                        else:
                            tau = 0.1
                    elif self.DoF == 5:
                        tau = 0.2
                    elif self.DoF == 2:
                        tau = 0.1
                    NK = self.reshapeNNDDtoNDND(contract('mn, ij -> mnij', kx, np.eye(self.DoF)))

                    H1 = self.H_posdef(Hmlpt, kx, gkx)
                    # H1 = self.H_bar(Hmlpt, kx, gkx)
                    H = H1 + NK * tau
                    UH = scipy.linalg.cholesky(H)

                    v_svgd = self.getSVGD_direction(kx, gkx, gmlpt_new)
                    v_svn = self.get_v_svn(kx, v_svgd, UH)
                    v_stc = self.get_v_stc(kx, UH)
                    update = (v_svn) * eps + v_stc * np.sqrt(eps)
                    # update = v_svn * eps
                elif method == 'sSVNintrem':
                    # v_drift = self.get_v_drift(kx, gkx, UH, A_nndd)
                    # Perform update
                    # update = (v_svn + v_drift) * eps + v_stc * np.sqrt(eps)
                    # update = (v_svn + v_drift) * eps
                    # update = (v_svn) * eps
                    # if iter_ < 5:
                    #     alphas = scipy.sparse.linalg.cg(H, v_svgd.flatten(), maxiter=5)[0].reshape(self.nParticles, self.DoF)
                    #     v_svn = contract('mn, ni -> mi', kx, alphas)
                    #     # update = v_svn * eps
                    # else:

                    # H = H1 + np.diag(np.abs(H1[range(self.dim), range(self.dim)])) @ NK * tau
                    # log.info('DEBUG: Rank of K: %f' % (np.linalg.matrix_rank(kx) * self.DoF))
                    # log.info('DEBUG: Determinant of kx: %.2e' % np.linalg.det(kx))
                    # Ensure K will be PD
                    # alpha, L_kx = self.getMinimumPerturbationCholesky(kx)
                    # if alpha != 0:
                    #     kx += alpha * np.eye(self.nParticles)
                    # A_nndd = self.getA_nndd(kx, UH)
                    # Calculate velocity field contributions
                    # if iter_ % 50 == 0:
                    #     eps *= .75
                    # log.info('DEBUG: Determinant of new kx: %.2e' % np.linalg.det(kx))

                    # min_eigval_kx1 = np.min(np.linalg.eigvalsh(kx))
                    # log.info('ALGORITHM: Minimum eigenvalue of kx: %.2e' % min_eigval_kx1)

                    # Testing to see if eigenvalues of SVN-Hessian H are the problem
                    # min_eigval = np.min(scipy.linalg.eigvalsh(H))
                    # log.info('ALGORITHM: Minimum eigenvalue of H: %.2e' % min_eigval)

                    # Testing to see if K H^{-1} H is the problem
                    # K = self.reshapeNNDDtoNDND(contract('mn, ij -> mnij', kx / self.nParticles, np.eye(self.DoF)))
                    # A = tf.linalg.cholesky_solve(UH.T, K).numpy()
                    # mat = K @ A
                    # log.info('DEBUG: K H^{-1} K max eigval: %.2e' % np.max(np.linalg.eigvalsh(mat)))
                    # log.info('DEBUG: H max eigval: %.2e' % np.max(np.linalg.eigvalsh(H)))
                    # eigvals_mat = np.linalg.eigvalsh(mat)
                    # eigvals_hposdef = np.linalg.eigvalsh(H)

                    # log.info('DEBUG: Rank of K H^{-1} K (With cholesky): %.f' % np.linalg.matrix_rank(mat))
                    # min_eigval = np.min(scipy.linalg.eigvalsh(mat))
                    # log.info('ALGORITHM: Minimum eigenvalue of D_SVN (With cholesky): %.2e' % min_eigval)
                    #
                    # mat1 = K @ np.linalg.inv(H) @ K
                    # min_eigval1 = np.min(scipy.linalg.eigvalsh(mat1))
                    # log.info('DEBUG: Rank of D_SVN (Direct): %.f' % np.linalg.matrix_rank(mat1))
                    # log.info('ALGORITHM: Minimum eigenvalue of D_SVN (Direct): %.2e' % min_eigval1)


                    # min_eigval_kx1 = np.min(np.linalg.eigvalsh(kx))
                    # log.info('MIN KX EIGENVALUE: %.2e' % min_eigval_kx1)


                    # min_eigval_kx = np.min(np.linalg.eigvalsh(kx))
                    # if min_eigval_kx < 0:
                    #     kx += np.abs(min_eigval_kx) * np.eye(self.nParticles)
                    # min_eigval_kx1 = np.min(np.linalg.eigvalsh(kx))
                    # log.info('KX MIN EIGVAL: %.2e' % min_eigval_kx1)

                    # eigvals = np.linalg.eigvalsh(H)
                    # norm1 = np.linalg.norm(v_svn.flatten() * eps)
                    # norm2 = np.linalg.norm(v_drift.flatten() * eps)
                    # norm3 = np.linalg.norm(v_stc.flatten() * np.sqrt(eps))
                    # log.info('||v_det|| = %.2e, ||v_drift|| = %.2e, ||v_stc|| = %.2e' % (norm1, norm2, norm3))
                    # if iter_ < 0: ### CG Solve ###
                    #     Hop = tf.linalg.LinearOperatorFullMatrix(H, is_self_adjoint=True, is_positive_definite=True) # 50 iters works for 10D!
                    #     alphas = tf.linalg.experimental.conjugate_gradient(Hop, tf.constant(v_svgd.flatten()), max_iter=100)[1].numpy().reshape(self.nParticles, self.DoF)
                    #     v_svn = contract('mn, ni -> mi', kx, alphas)
                    #     update = v_svn * eps
                    # else: ### Cholesky solve ###
                    if iter_ > 20:
                        eps = 0.1

                    GN_Hmlpt = self.getGNHessianMinusLogPosterior_ensemble_new(X_new)
                    self.check_rank_GN(GN_Hmlpt)
                    kx, gkx = getKernelWithDerivatives(X_new, h=h, M=np.mean(GN_Hmlpt, axis=0))
                    # kx, gkx = getKernelWithDerivatives(X_new, h=h)
                    v_svgd = self.getSVGD_direction(kx, gkx, gmlpt_new)
                    H = self.H_posdef(GN_Hmlpt, kx, gkx)
                    UH = scipy.linalg.cholesky(H) # Upper
                    # Hop = tf.linalg.LinearOperatorFullMatrix(H, is_self_adjoint=True, is_positive_definite=True) # 50 iters works for 10D!
                    B = np.random.normal(0, 1, self.dim)
                    tmp1 = v_svgd.flatten() * eps + np.sqrt(2 / self.nParticles) * UH.T @ B * np.sqrt(eps) * 0
                    tmp2 = scipy.linalg.cho_solve((UH, False), tmp1.flatten()).reshape(self.nParticles, self.DoF)
                    update = contract('mn, ni -> mi', kx, tmp2)
                    # update = tf.linalg.experimental.conjugate_gradient(Hop, tf.constant(tmp1), max_iter=10)[1].numpy().reshape(self.nParticles, self.DoF)


                elif method == 'sSVNfail':
                    Hmlpt_new = self.getGNHessianMinusLogPosterior_ensemble_new(X_new)
                    kx, gkx = getKernelWithDerivatives(X_new, h=h, M=np.mean(Hmlpt_new, axis=0))
                    kx_sqrt, alpha = self.compute_sqrt_if_possible(kx)
                    if alpha != 0:
                        kx += alpha * np.eye(self.nParticles)
                    v_svgd = self.getSVGD_direction(kx, gkx, gmlpt_new)
                    HBD = self.h_ij_BD(Hmlpt_new, kx, gkx)
                    LHBD = tf.linalg.cholesky(HBD).numpy()
                    if iter_ < 5:
                        HBDop = tf.linalg.LinearOperatorFullMatrix(HBD, is_self_adjoint=True, is_positive_definite=True)
                        v_svn = tf.linalg.experimental.conjugate_gradient(HBDop, tf.constant(v_svgd), max_iter=1).x.numpy()
                    else:
                        v_svn = np.squeeze(tf.linalg.cholesky_solve(LHBD, v_svgd[..., np.newaxis])) # agreed
                    v_stc = self.getSVN_vstc(LHBD, kx_sqrt)
                    update = v_svn * eps + v_stc * np.sqrt(eps)

                # For diagnostic purposes
                h_med = self.bandwidth_MED(X_new)
                log.info('ALGORITHM: MED_BW = %f, NEW_BW = %f' % (h_med, h))
                log.info('ALGORITHM: Output step = %f, Output bandwidth = %f' % (eps, h))

                ###########################################################
                # Store relevant per iteration information
                ###########################################################
                with h5py.File(self.history_path_new, 'a') as f:
                    g = f.create_group('%i' % iter_)
                    g.create_dataset('X', data=copy.deepcopy(X_new))
                    g.create_dataset('gmlpt', data=copy.deepcopy(gmlpt_new))

                X_new += update
                # end for loop

            ###################################################
            # Store metadata for future reference
            ###################################################
            with h5py.File(self.history_path_new, 'a') as f:
                g = f.create_group('metadata')
                g.create_dataset('nLikelihoodEvals', data=copy.deepcopy(self.model.nLikelihoodEvaluations))
                g.create_dataset('nGradLikelihoodEvals', data=copy.deepcopy(self.model.nGradLikelihoodEvaluations))
                g.create_dataset('nParticles', data=copy.deepcopy(self.nParticles))
                g.create_dataset('DoF', data=copy.deepcopy(self.DoF))
                g.create_dataset('L', data=copy.deepcopy(iter_ + 1))
                g1 = f.create_group('final_updated_particles')
                g1.create_dataset('X', data=X_new)

        #################################################################################
        # Save profiler results to HTML
        #################################################################################
        if self.profile == True:
            log.info('OUTPUT: Saving profile as html')
            profiler.stop()
            try:
                with open(os.path.join(self.RUN_OUTPUT_DIR, 'output.html'), "w") as f:
                    profile_html = profiler.output_html()
                    log.info(type(profile_html))
                    f.write(profile_html)
                    log.info('OUTPUT: Successfully saved profile to html.')
                    # log.info(profiler.output_text(unicode=True, color=True))
            except:
                log.error('OUTPUT: Failed to save profile to html. Trying utf-8', exc_info=True)
                try:
                    with open(os.path.join(self.RUN_OUTPUT_DIR, 'output.html'), "wb") as f:
                        f.write(profile_html.encode('utf-8'))
                        log.info('OUTPUT: Successfully saved profile to html.')
                except:
                    log.error('OUTPUT: Failed to save utf-8 profile to html.', exc_info=True)

        log.info('OUTPUT: Run completed successfully! Data stored in:\n %s' % self.history_path_new)
        return_dict = {'path_to_results': self.history_path_new, 'outdir_path': self.RUN_OUTPUT_DIR}
        return return_dict

    except Exception as e: log.error(e)

##############################################################
# bandwidth selection
##############################################################
def getBandwidth_new(self, X, h_guess, M):
    tmp1 = self.DoF ** 2 / (np.trace(M) * self.nParticles) ** 2
    def cost_HE(h):
        kx, gkx, tr_hesskx = getKernelWithDerivatives(X, h=h, M=M, get_tr_hesskx=True)
        score_estimate = np.sum(gkx, axis=1) / np.sum(kx, axis=1).reshape(self.nParticles, 1)
        scale = tmp1 * h ** 2
        r = np.sum(tr_hesskx, axis=1) - contract('mni, ni -> m', gkx, score_estimate)
        return np.sum(r ** 2) * scale
    # return scipy.optimize.minimize(cost_HE, bounds=[(h_guess/5, h_guess*5)], method='L-BFGS-B', x0=h_guess)['x']
    # return scipy.optimize.minimize(cost_HE, bounds=(h_guess/5, h_guess*5), method='CG', x0=h_guess)['x']
    # pass
    # def cost_BM(h):
    #     tau = 0.01
    #     kx, gkx = getKernelWithDerivatives(X, h=h, M=M)
    #     score_estimate = np.sum(gkx, axis=1) / np.sum(kx, axis=1).reshape(self.nParticles, 1)
    #     B = np.random.normal(0, 1, (self.nParticles, self.DoF))
    #     return MMD(X + np.sqrt(2 * tau) * B, X - tau * score_estimate)
    # DEBUGGING
    # import matplotlib.pyplot as plt
    # array = np.arange(0.0001, h_guess*5, 0.01)
    # plt.plot(array, (np.vectorize(cost_HE))(array))
    # plt.show()
    return scipy.optimize.minimize_scalar(cost_HE, bounds=(h_guess/5, h_guess + 50), method='bounded')['x']
    # return self.optimize_quadratic_interpolation(h_guess, cost_HE)
def optimize_quadratic_interpolation(self, h0, cost):
    # Adapted from (https://github.com/YiifeiWang/Accelerated-Information-Gradient-flow/blob/master/utils/BM_bandwidth.m)
    explore_ratio = 1.1
    cost0 = cost(h0)
    eps = 1e-6
    # Get gradient sign
    gCost0 = (cost(h0 + eps) - cost0) / eps
    # Step in decreasing direction
    if gCost0 < 0:
        h1 = h0 * explore_ratio
    else:
        h1 = h0 / explore_ratio
    # Make sure h1 is valid
    if h1 < 0:
        return h0
    cost1 = cost(h1)
    s = (cost1 - cost0) / (h1 - h0)
    h2 = (h0 * s - 0.5 * gCost0 * (h1 + h0)) / (s - gCost0)
    # Make sure h2 is valid
    if h2 < 0:
        if cost1 < cost0:
            return h1
        else:
            return h0
    cost2 = cost(h2)
    log.info('BANDWIDTH: h0 = %f, h1 = %f, h2 = %f' % (h0, h1, h2))
    log.info('BANDWIDTH: cost0 = %f, cost1 = %f, cost2 = %f' % (cost0, cost1, cost2))
    if cost0 < cost1 and cost0 < cost2:
        return h0
    if cost1 < cost0 and cost1 < cost2:
        return h1
    if cost2 < cost0 and cost2 < cost1:
        return h2

#################################################
# solver
#################################################

# def solveSystem_new(self, H_bar, mgJ):
#     # maxiter = 100
#     # maxiter = 1000
#     maxiter = 50
#     self.callback_reset()
#     start = time()
#     # res = scipy.sparse.linalg.gmres(H_bar, mgJ, atol=1e-2, maxiter=maxiter, callback=self.callback)[0].reshape(self.nParticles, self.DoF)
#
#     # jitter, chol = self.getMinimumPerturbationCholesky(H_bar)
#
#     # res = scipy.linalg.cho_solve((chol, True), mgJ).reshape(self.nParticles, self.DoF)
#     # res = scipy.linalg.solve(H_bar, mgJ, assume_a='gen').reshape(self.nParticles, self.DoF)
#     # res = scipy.sparse.linalg.cg(H_bar, mgJ, maxiter=5, callback=self.callback)[0].reshape(self.nParticles, self.DoF)
#     res = scipy.sparse.linalg.cg(H_bar, mgJ, maxiter=10, callback=self.callback)[0].reshape(self.nParticles, self.DoF)
#     # res = scipy.sparse.linalg.gmres(H_bar, mgJ, maxiter=500, callback=self.callback)[0].reshape(self.nParticles, self.DoF)
#     # res = scipy.sparse.linalg.minres(H_bar, mgJ, maxiter=500, callback=self.callback)[0].reshape(self.nParticles, self.DoF)
#     # res = scipy.sparse.linalg.gcrotmk(H_bar, mgJ, maxiter=10, callback=self.callback)[0].reshape(self.nParticles, self.DoF)
#     end = time()
#     total_time = end - start
#     log.info('SOLVE: Iterations performed: %i' % self.cgiter)
#     log.info('SOLVE: Time to solution: %f' % (total_time))
#     return res
#
# def callback_reset(self):
#     # for cg iter debug
#     self.cgiter=0
#     self.dict_cg={}
#
# def callback(self, xk):
#     # for cg iter debug
#     x = np.copy(xk)
#     self.cgiter += 1
#     self.dict_cg[self.cgiter] = x
#
#############################################
# extras
#############################################

# def check_rank_GN(self, GN_Hmlpt):
#     ranks = tf.linalg.matrix_rank(GN_Hmlpt).numpy()
#     average_rank = np.mean(ranks)
#     log.info('Expected rank: %i, Average rank: %.2f' % (self.DoF, average_rank))

    def pushForward(self, samples, history_path):
        T = copy.deepcopy(samples)
        with h5py.File(history_path, 'r') as f:
            num_iter = f['metadata'].attrs.__getitem__('total_num_iterations')
            method = f['metadata'].attrs.__getitem__('optimization_method')
            for l in range(num_iter):
                #################################################
                # Loaded data from h5
                #################################################
                eps = f['%i' % l]['eps'][()]
                M = f['%i' % l]['M'][...]
                h = f['%i' % l]['h'][()]
                X = f['%i' % l]['X'][...]
                gmlpt = f['%i' % l]['gmlpt'][...]
                #################################################
                # Recalculate using stored data.
                #################################################
                deltas = self.getDeltas(X, X)
                metricDeltas = self.getMetricDeltas(M, deltas)
                deltaMetricDeltas = self.getDeltasMetricDeltas(deltas, metricDeltas)
                kx = self.getKernelPiecewise(h, deltaMetricDeltas)
                gkx = self.getGradKernelPiecewise(h, kx, metricDeltas)
                # mgJ = self.mgJ_new(kx, gkx, gmlpt)
                ###############################################
                # Match recalculated and stored data
                ###############################################
                # t0 = self.check_if_same(h, self.DEBUG_dict[l]['h'])
                # t1 = self.check_if_same(X, self.DEBUG_dict[l]['X'])
                # t2 = self.check_if_same(deltas, self.DEBUG_dict[l]['deltas'])
                # t3 = self.check_if_same(metricDeltas, self.DEBUG_dict[l]['metricDeltas'])
                # t4 = self.check_if_same(deltaMetricDeltas, self.DEBUG_dict[l]['deltasMetricDeltas'])
                # t5 = self.check_if_same(kx, self.DEBUG_dict[l]['kx'])
                # t6 = self.check_if_same(gkx, self.DEBUG_dict[l]['gkx'])
                # t7 = self.check_if_same(mgJ, self.DEBUG_dict[l]['mgJ'])
                ###############################################
                # Calculate kt and gkt (Part of final algorithm)
                ###############################################
                deltas_T = self.getDeltas(X, T)
                metricDeltas_T = self.getMetricDeltas(M, deltas_T)
                deltaMetricDeltas_T = self.getDeltasMetricDeltas(deltas_T, metricDeltas_T)
                kt = self.getKernelPiecewise(h, deltaMetricDeltas_T)
                gkt = self.getGradKernelPiecewise(h, kt, metricDeltas_T)
                ###############################################
                # Construct update
                ###############################################
                if method == 'SVN':
                    alphas = f['%i' % l]['alphas'][...]
                    wt = self.w(alphas, kt)
                elif method == 'SVGD':
                    wt = self.mgJ_new(kt, gkt, gmlpt)
                ###############################################
                # For T = X0, these conditions should hold
                ###############################################
                # t8 = self.check_if_same(X, T)
                # t9 = self.check_if_same(deltas, deltas_T)
                # t10 = self.check_if_same(metricDeltas, metricDeltas_T)
                # t11 = self.check_if_same(deltaMetricDeltas, deltaMetricDeltas_T)
                # t12 = self.check_if_same(mgJ, wt)
                T += eps * wt
                pass
            return T


    def isPD(self, B):
        """Returns true when input is positive-definite, via Cholesky"""
        try:
            _ = la.cholesky(B)
            return True
        except la.LinAlgError:
            return False

    # def getMinimumPerturbationMatrixsqrt(self, x, jitter=1e-9):
    #     try:
    #         cholesky = scipy.linalg.sqrtm(x)
    #         return cholesky, 0
    #     except Exception:
    #         while jitter < 1.0:
    #         # while jitter < 2.0:
    #             try:
    #                 cholesky = scipy.linalg.sqrtm(x + jitter * np.eye(x.shape[0]))
    #                 return cholesky, jitter
    #             except Exception:
    #                 log.info('CHOLESKY: Matrix not positive-definite. Adding alpha = %f' % jitter)
    #                 jitter = jitter * 10
    #         raise Exception('CHOLESKY: Factorization failed.')

    def getSVN_vstc(self, LHBD, kx_sqrt, B=None):
        UHBD = contract('mij -> mji', LHBD)
        if B is None:
            B = np.random.normal(0, 1, (self.nParticles, self.DoF))
        # DEBUG
        tmp1 = np.zeros((self.nParticles, self.DoF))
        for n in range(self.nParticles):
            tmp1[n] = scipy.linalg.solve_triangular(UHBD[n], B[n])
        # The triangular solve method has bugs in it!!!
        # tmp1 = np.squeeze(tf.linalg.triangular_solve(UHBD, B[..., np.newaxis], lower='False')) #1
        tmp2 = tmp1.flatten(order='F').reshape(self.DoF, self.nParticles) #2
        tmp3 = contract('mn, in -> im', kx_sqrt, tmp2) #3
        tmp4 = tmp3.flatten(order='F').reshape(self.nParticles, self.DoF) #4
        return np.sqrt(2 / self.nParticles) * tmp4

    def checkGradients(self, X, gmlpt):
        rtol = 1e-6
        atol = 1e-6
        i = np.random.randint(0, self.DoF)
        random_particle = X[i]
        grad_method = gmlpt[i]
        grad_numerical = nd.Gradient(self.getMinusLogPosterior_individual)(random_particle)
        assert np.allclose(grad_method, grad_numerical, rtol=rtol, atol=atol)
        log.info('Gradients are calculated correctly and organized correctly.')
        print(grad_method, grad_numerical)

##############################################
# class LinearOperatorSVN_Hessian():
#     def __init__(self, operator, is_square=True, name=None):
#         parameters = dict(
#             operator=operator,
#             is_square=is_square,
#             name=name)
#         super().__init__(..., parameters=parameters)

# def main():
#     pass
# def load_logger(config_name):
#     """
#     Load configuration file for logger from same folder as driver
#     Args:
#         file_name: .ini file name
#
#     Returns: log object
#
#     """
#     directory = os.path.dirname(os.path.abspath(__file__))
#     logger_config_path = '%s/%s' % (directory, config_name)
#     assert os.path.exists(logger_config_path), 'Logger configuration file not found.'
#     logging.config.fileConfig(logger_config_path, disable_existing_loggers=False)
#     return logging.getLogger()

# if __name__ == '__main__':
#     # Setup argparse
#     parser = argparse.ArgumentParser(description='Driver for Stein Variational Inference Package')
#     parser.add_argument('-nIter', '--nIterations', type=int, required=True, help='Number of iterations')
#     parser.add_argument('-nP', '--nParticles', type=int, required=True, help='Number of particles')
#     parser.add_argument('-method', '--optimize_method', type=str, required=True, help='Choice of optimization')
#     parser.add_argument('-prf', '--profile', type=str, required=False, help='Output algorithm profile information')
#     args = parser.parse_args()
#
#     config_name = 'logger_configuration.ini'
#     log = load_logger(config_name)
#
#     log.info('Beginning job')
#     main()
#     log.info('Ending job')

######################################################################
# Hybrid rosenbrock
######################################################################
#################################################
# HRD forward evaluations
#################################################
# def getMinusLogLikelihood_individual(self, x_in):
#     x_graph = np.insert(x_in[1:].reshape(self.n2, self.n1-1), 0, x_in[0], axis=1)
#     r1 = np.sqrt(2 * self.a) * (x_in[0] - self.mu_rosen)
#     rji = np.sqrt(2 * self.b) * (x_graph[:, 1:] - x_graph[:, :-1] ** 2)
#     # return -np.log(self.z_inv) + r1 ** 2 / 2 + np.sum(rji ** 2) / 2 # With normalization
#     return r1 ** 2 / 2 + np.sum(rji ** 2) / 2 # With normalization
#     # return (r1 ** 2 + np.sum(rji ** 2)) / 2
#
# def getGradientMinusLogLikelihood_individual(self, x_in):
#     x_graph = np.insert(x_in[1:].reshape(self.n2, self.n1-1), 0, x_in[0], axis=1)
#     r1 = np.sqrt(2 * self.a) * (x_in[0] - self.mu_rosen)
#     rji = np.sqrt(2 * self.b) * (x_graph[:, 1:] - x_graph[:, :-1] ** 2)
#     tmp1 = np.einsum('ji, jie -> jie', np.sqrt(2 * self.b), self.mat['delta1'])
#     tmp2 = -1 * np.einsum('ji, ji, jie -> jie', np.sqrt(2 * self.b), 2 * x_graph[:, :-1], self.mat['delta2'])
#     grji = tmp1 + tmp2
#     return r1 * self.gr1 + np.einsum('ji, jie -> e', rji, grji)
#
# def getHessianMinusLogLikelihood_individual(self, x_in):
#     x_graph = np.insert(x_in[1:].reshape(self.n2, self.n1-1), 0, x_in[0], axis=1)
#     # r1 = np.sqrt(2 * self.a) * (x_in[0] - self.mu_rosen)
#     rji = np.sqrt(2 * self.b) * (x_graph[:, 1:] - x_graph[:, :-1] ** 2)
#     tmp1 = np.einsum('ji, jie -> jie', np.sqrt(2 * self.b), self.mat['delta1'])
#     tmp2 = -1 * np.einsum('ji, ji, jie -> jie', np.sqrt(2 * self.b), 2 * x_graph[:, :-1], self.mat['delta2'])
#     grji = tmp1 + tmp2
#     return np.einsum('f,e -> ef', self.gr1, self.gr1) + \
#             np.einsum('jif, jie -> ef', grji, grji) + \
#             np.einsum('ji, jief -> ef', rji, self.Hrji)
#
# def getGNHessianMinusLogLikelihood_individual(self, x_in):
#     x_graph = np.insert(x_in[1:].reshape(self.n2, self.n1-1), 0, x_in[0], axis=1)
#     # r1 = np.sqrt(2 * self.a) * (x_in[0] - self.mu_rosen)
#     # rji = np.sqrt(2 * self.b) * (x_graph[:, 1:] - x_graph[:, :-1] ** 2)
#     tmp1 = np.einsum('ji, jie -> jie', np.sqrt(2 * self.b), self.mat['delta1'])
#     tmp2 = -1 * np.einsum('ji, ji, jie -> jie', np.sqrt(2 * self.b), 2 * x_graph[:, :-1], self.mat['delta2'])
#     grji = tmp1 + tmp2
#     return np.einsum('f,e -> ef', self.gr1, self.gr1) + \
#            np.einsum('jif, jie -> ef', grji, grji)
#
# def getGradientGNHessianMinusLogLikelihood_individual(self, x_in):
#     x_graph = np.insert(x_in[1:].reshape(self.n2, self.n1-1), 0, x_in[0], axis=1)
#     # r1 = np.sqrt(2 * self.a) * (x_in[0] - self.mu_rosen)
#     # rji = np.sqrt(2 * self.b) * (x_graph[:, 1:] - x_graph[:, :-1] ** 2)
#     tmp1 = np.einsum('ji, jie -> jie', np.sqrt(2 * self.b), self.mat['delta1'])
#     tmp2 = -1 * np.einsum('ji, ji, jie -> jie', np.sqrt(2 * self.b), 2 * x_graph[:, :-1], self.mat['delta2'])
#     grji = tmp1 + tmp2
#     return np.einsum('jifg, jie -> efg', self.Hrji, grji) + \
#             np.einsum('jif, jieg -> efg', grji, self.Hrji)

##########################################
# Draw samples
##########################################
# def newDrawFromHRD(self, N):
#     samples = np.zeros((N, self.DoF))
#     # Populate first dimension of all samples
#     samples[:, 0] = np.random.normal(self.mu_rosen, 1 / (np.sqrt(2 * self.a)), N)
#     v = lambda n: n-1
#     for k in range(N):
#         # Added sqrt
#         samples[k, 0] = np.random.normal(self.mu_rosen, 1 / (2 * self.a))
#         # Iteratively build up sample using very first draw
#         for m, n in itertools.product(range(self.n2), range(self.n1-1)):
#             # Added sqrt
#             samples[k, self.mu(m,n+1)] = np.random.normal(samples[k, self.mu(m,n)] ** 2, 1 / (2 * self.b[m, v(n+1)]))
#     return samples
# def newDrawFromLikelihood_old(self, N):
#     samples = np.zeros((N, self.DoF))
#     # Populate first dimension of all samples
#     samples[:, 0] = np.random.normal(self.mu_rosen, 1 / (np.sqrt(2 * self.a)), N)
#     for k in range(N):
#         for j, I in itertools.product(range(self.n2), range(self.n1-1)):
#             samples[k, self.mu(j,I+1)] = np.random.normal(samples[k, self.mu(j,I)] ** 2, 1 / np.sqrt(2 * self.b[j,I]))
#     return samples


    def setup_rosenbrock(self):
        # Calculate input dependent quantities
        if self.id is None:
            self.modelType = 'hybrid_rosenbrock_%i_%i' % (self.n2, self.n1)
        else:
            self.modelType = 'hybrid_rosenbrock_%i_%i_%s' % (self.n2, self.n1, self.id)
        self.z_inv = self.getNormalizationConstant()
        self.mat = self.formKroneckerDeltas()
        # self.mat = self.formKroneckerMatricies()
        # self.x_graph_index = self.getxGraph_index()

        # Calculate static residuals
        self.gr1 = np.zeros(self.DoF)
        self.gr1[0] = np.sqrt(2 * self.a)
        self.Hrji = -1 * np.einsum('ji, jif, jie -> jief', np.sqrt(8 * self.b), self.mat['delta2'], self.mat['delta2'])

    def formKroneckerDeltas(self):
        def index(j, i):
            # Given position in graph, return corresponding component of input vector
            if i == 0:
                return int(0)
            elif i > 0:
                return int(j * (self.n1 - 1) + i)
        delta = np.zeros((self.n2, self.n1, self.DoF))
        for j, i in itertools.product(range(self.n2), range(self.n1)):
            delta[j, i, index(j, i)] = 1
        return {'delta1': delta[:, 1:], 'delta2': delta[:, :-1]}



        #     # Added sqrt
        #     samples[k, 0] = np.random.normal(self.mu_rosen, 1 / (2 * self.a))
        #     # Iteratively build up sample using very first draw
        #     for m, n in itertools.product(range(self.n2), range(self.n1-1)):
        #         # Added sqrt
        #         samples[k, self.mu(m,n+1)] = np.random.normal(samples[k, self.mu(m,n)] ** 2, 1 / (2 * self.b[m, v(n+1)]))
        # return samples



    # def newDrawFromLikelihood(self, N):
    #     samples = np.zeros((N, self.DoF))
    #     for d in range(self.DoF):
    #         if d == 0:
    #             samples[:, 0] = np.random.normal(self.mu_rosen, 1 / (np.sqrt(2 * self.a)), N)
    #         else:
    #             samples[:,d] = np.random.normal(samples[:,d-1] ** 2, 1 / (np.sqrt(2 * self.b[0,0])), N) # assume b constant
    #     return samples


        # Set parameters for prior
        self.sigma = np.eye(self.DoF)
        self.sigma_inv = np.linalg.inv(self.sigma)
        # self.mu_gauss = np.zeros(self.DoF)
        # self.mu_gauss = -1 * np.ones(self.DoF)
        self.mu_gauss = np.ones(self.DoF)

####################################
# reshaping methods for N x N blocks
####################################

    def _reshapeNDNDtoDDNN(self, H):
        block_shape = np.array((self.nParticles, self.nParticles))
        new_shape = tuple(H.shape // block_shape) + tuple(block_shape)
        new_strides = tuple(H.strides * block_shape) + H.strides
        return np.lib.stride_tricks.as_strided(H, shape=new_shape, strides=new_strides)