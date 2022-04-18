import numpy as np
import itertools
from models.rosenbrock_analytic_proper_sample import rosenbrock_analytic
from source.stein_experiments import SVI
import os
import copy
import scipy
import tensorflow as tf
# from tensorflow.linalg.sparse import conjugate_gradient
import numdifftools as nd
import matplotlib.pyplot as plt
import opt_einsum as oe
from time import time, sleep
# t = timeit.Timer(functools.partial(self.stein.getGNHessianMinusLogPosterior_ensemble_new, self.stein.particles))
# t_contract = timeit.Timer(functools.partial(self.H_bar_contract, GN_Hmlpt, kernel, gradKernel))
class Test_sanity:
    def setup(self):
        # output_dir = os.path.dirname(os.path.abspath(__file__)) + '/outdir'
        model = rosenbrock_analytic()
        ###########################################################
        # Define and run for 1 iteration to instantiate all methods
        ###########################################################
        self.rtol = 1e-15
        self.atol = 1e-15
        self.optimizeMethod = 'SVN'
        self.nParticles = 4
        # self.nParticles = 5
        self.nIterations = 1
        self.stein = SVI(model = model, nParticles=self.nParticles, nIterations=self.nIterations, optimizeMethod=self.optimizeMethod)
        self.stein.SVGD_stochastic_correction = False
        self.nParticles_test = self.nParticles
        self.test_set_history = {}
        # np.random.seed(int(time()))
        np.random.seed(1)
        self.X = np.copy(self.stein.model.newDrawFromPrior(self.nParticles)) # copy because particles are updated after apply()
        self.DoF = self.stein.DoF
        self.stein.pickleData = False
        self.stein.constructMap()

        # Define objects for convenience
        self.dim = self.nParticles * self.DoF
        self.delta_dim = np.eye(self.dim)
        self.delta_N = np.eye(self.nParticles)
        self.delta_D = np.eye(self.DoF)
        ##############################################################
        # Calculations: Vanilla SVN
        ##############################################################
        self.M = np.eye(self.DoF)
        self.deltas = self.stein.getDeltas(self.X, self.X)
        self.metricDeltas = self.stein.getMetricDeltas(self.M, self.deltas)
        self.deltasMetricDeltas = self.stein.getDeltasMetricDeltas(self.deltas, self.metricDeltas)
        self.bandwidth = self.stein.getBandwidth(self.deltasMetricDeltas)
        self.kx = self.stein.getKernelPiecewise(self.bandwidth, self.deltasMetricDeltas)
        self.gkx = self.stein.getGradKernelPiecewise(self.bandwidth, self.kx, self.metricDeltas)
        self.Hesskx = self.stein.getHessianKernel(self.bandwidth, self.kx, self.gkx, self.M, self.metricDeltas)
        self.gmlpt = self.stein.getGradientMinusLogPosterior_ensemble_new(self.X)
        self.GN_Hmlpt = self.stein.getGNHessianMinusLogPosterior_ensemble_new(self.X)
        self.H_bar = self.stein.H_bar(self.GN_Hmlpt, self.kx, self.gkx)
        ##############################################################
        # Calculations: Full stochastic SVN
        ##############################################################
        # /// Form all matricies (ND x ND)
        self.K = self.formK()
        self.D_K = self.form_D_K() # SVGD diffusion matrix
        self.gradK = self.form_gradK_first_slot() + self.form_gradK_second_slot()
        self.divK = np.einsum('abb -> a', self.gradK)
        self.gradH_numerical = np.einsum('ijk -> kij', nd.Gradient(self.form_H_for_numdiff)(self.X.flatten()))

        cholesky_dict = self.stein.compute_cholesky_if_possible(self.H_bar, 1e-9)
        self.alpha = cholesky_dict['alpha']
        self.H_PD_ndnd = self.H_bar + self.alpha * np.eye(self.dim)
        self.H_inv_ndnd = np.linalg.inv(self.H_PD_ndnd)
        self.H_inv_nndd = self.stein.reshapeNDNDtoNNDD(self.H_inv_ndnd)
        self.A_ndnd = self.H_inv_ndnd @ self.K
        self.A_nndd = self.stein.reshapeNDNDtoNNDD(self.A_ndnd)
        self.D_SVN = self.nParticles * self.K @ self.H_inv_ndnd @ self.K
        self.phi = lambda m, d: m * self.DoF + d
        self.grad_GN_Hmlpt = self.stein.getGradientGNHessianMinusLogPosterior_ensemble(self.X)
        ##############################################################
        # Calculations: Block diagonal stochastic SVN
        ##############################################################
        self.HBD = self.stein.H_bar_BD(self.GN_Hmlpt, self.kx, self.gkx)
        self.HBD_inv = tf.linalg.inv(self.HBD).numpy()

        self.K_HBD_inv = self.stein.K_action_mbd(self.kx, self.HBD_inv)
        self.HBD_inv_K = np.einsum('mnbd -> nmdb', self.K_HBD_inv)

        self.HBD_inv_ndnd = self.makeFullMatrixFromBlockDiagonal(self.HBD_inv)
        self.D_SVN_BD = self.nParticles * self.K @ self.HBD_inv_ndnd @ self.K # SVN diffusion matrix
        self.L_ndd = tf.linalg.cholesky(self.HBD_inv).numpy()
        self.L_ndnd = self.makeFullMatrixFromBlockDiagonal(self.L_ndd)
        self.A_BD_nndd = np.einsum('mij, mn -> mnij', self.HBD_inv, self.kx) / self.nParticles
        # Test that this agrees
        self.A_BD_nndd = np.einsum('no, nje -> noje', self.kx, self.HBD_inv) / self.nParticles

        self.A_BD_ndnd = self.stein.reshapeNNDDtoNDND(self.A_BD_nndd)
        self.grad_hij_BD = self.stein.grad_hij_BD_TESTING(self.kx, self.gkx, self.Hesskx, self.GN_Hmlpt, self.grad_GN_Hmlpt)
        # /// Form matricies
        self.gradHBD = self.form_grad_hij_BD()
        self.gradHBD_numerical = nd.Gradient(self.form_HBD_for_numerical_derivative)(self.X.flatten())
        self.gradHBD_numerical = np.einsum('ijk -> kij', self.gradHBD_numerical)
        ##############################################################
        # Calculations: N x N Block diagonal algorithm
        ##############################################################
        self.HBD_NxN = self.stein.HBD_NxN(self.kx, self.gkx, self.GN_Hmlpt)



    ################################################################
    # New NxN methods
    ################################################################
# operator = tf.linalg.LinearOperatorFullMatrix(self.makeFullMatrixFromBlockDiagonal(HBD), is_self_adjoint=True, is_positive_definite=True)
# K = self.reshapeNNDDtoNDND(np.einsum('mn, ij -> mnij', kx, np.eye(self.DoF))) / self.nParticles
# try:
#     output_cg = tf.linalg.experimental.conjugate_gradient(operator, tf.constant(K), max_iter=20)
# except:
#     operator = tf.linalg.LinearOperatorFullMatrix(self.makeFullMatrixFromBlockDiagonal(HBD) + 1e-3 * np.eye(self.dim), is_self_adjoint=True, is_positive_definite=True)
#     output_cg = tf.linalg.experimental.conjugate_gradient(operator, tf.constant(K), max_iter=20)
# result = output_cg[1].numpy().T
# log.info('INFO: Number of CG iterations %i' % output_cg[0])
# A = self.reshapeNDNDtoNNDD(result)
    def test_CG_solver(self):
        tol = 1e-6
        operator = tf.linalg.LinearOperatorFullMatrix(self.HBD_NxN[0], is_self_adjoint=True, is_positive_definite=True)
        kx = tf.constant(self.kx)
        result = tf.linalg.experimental.conjugate_gradient(operator, kx, tol=tol)[1].numpy().T

        res1 = np.zeros((self.nParticles, self.nParticles))
        # res2 = np.zeros((self.nParticles, self.nParticles))
        for n in range(self.nParticles):
            res1[:, n] = scipy.sparse.linalg.cg(self.HBD_NxN[0], kx[n], tol=tol)[0]
            # res2[n] = scipy.sparse.linalg.cg(self.HBD_NxN[0], kx[n], tol=tol)[0]
        pass


        # res =
        # np.apply_along_axis
        # scipy.sparse.linalg.cg()
        #
        # tf.linalg.LinearOperator(np.float64, graph_parents=None, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None, is_square=None, name=None, parameters=None)
        # tf.linalg.LinearOperator.a
        #
        #
        # # tf.linalg.sparse.conjugate_gradient
        # kx = tf.constant(self.kx)
        # HBD0 = tf.constant(self.HBD_NxN[0])
        # operator = tf.linalg.LinearOperator(HBD0)
        # # operator = tf.linalg.LinearOperatorFullMatrix(self.HBD_NxN[0])
        # tf.linalg.experimental.conjugate_gradient(HBD0, kx)
        # pass
    ############################################################
    # Stochastic SVN: Helper methods
    ############################################################
    def form_H_for_numdiff(self, X):
        # Input is an m*b vector
        X = X.reshape(self.nParticles, self.DoF)
        deltas = self.stein.getDeltas(X, X)
        metricDeltas = self.stein.getMetricDeltas(self.M, deltas)
        deltaMetricDeltas = self.stein.getDeltasMetricDeltas(deltas, metricDeltas)
        kx = self.stein.getKernelPiecewise(self.bandwidth, deltaMetricDeltas)
        gkx = self.stein.getGradKernelPiecewise(self.bandwidth, kx, metricDeltas)
        GN_Hmlpt = self.stein.getGNHessianMinusLogPosterior_ensemble_new(X)
        H = self.stein.H_bar(GN_Hmlpt, kx, gkx)
        return H


    def form_HBD_for_numerical_derivative(self, X):
        # Input is an m*b vector
        X = X.reshape(self.nParticles, self.DoF)
        deltas = self.stein.getDeltas(X, X)
        metricDeltas = self.stein.getMetricDeltas(self.M, deltas)
        deltaMetricDeltas = self.stein.getDeltasMetricDeltas(deltas, metricDeltas)
        kx = self.stein.getKernelPiecewise(self.bandwidth, deltaMetricDeltas)
        gkx = self.stein.getGradKernelPiecewise(self.bandwidth, kx, metricDeltas)
        GN_Hmlpt = self.stein.getGNHessianMinusLogPosterior_ensemble_new(X)
        h_ij_BD = self.stein.h_ij_BD(GN_Hmlpt, kx, gkx)
        H_BD = self.makeFullMatrixFromBlockDiagonal(h_ij_BD)
        return H_BD
    #
    def form_grad_hij_BD(self):
        # Form dim ** 3 tensor from mnije tensor (figure in paper)
        res = np.zeros((self.dim, self.dim, self.dim))
        for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
            block = self.grad_hij_BD[m,n]
            res[m * self.DoF : self.DoF * (m + 1), m * self.DoF : self.DoF * (m + 1), n * self.DoF : self.DoF * (n + 1)] = block
        return res

    def formK(self):
        # Form SVGD diffusion matrix K manually
        dim = self.nParticles * self.DoF
        res = np.zeros((dim, dim))
        for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
            block = self.kx[m, n] * np.eye(self.DoF)
            res[m * self.DoF : self.DoF * (m + 1), n * self.DoF : self.DoF * (n + 1)] = block
        return res / self.nParticles

    def form_gradK_first_slot(self):
        # Tensor contribution to gradK from first kernel slot
        dim = self.nParticles * self.DoF
        res = np.zeros((dim, dim, dim))
        # Get the gradient of augmented kernel matrix
        for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
            zeros = np.zeros((self.nParticles, self.DoF))
            zeros[m] = copy.deepcopy(self.gkx[m, n])
            zeros = zeros.flatten()
            block = np.einsum('ij, z -> ijz', np.eye(self.DoF), zeros)
            res[m * self.DoF : self.DoF * (m + 1), n * self.DoF : self.DoF * (n + 1), :] = block
        res /= self.nParticles
        return res

    def form_gradK_second_slot(self):
        # Tensor contribution to gradK from second slot + evidence that only second slot contributes to divergence
        dim = self.nParticles * self.DoF
        res = np.zeros((dim, dim, dim))
        for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
            zeros = np.zeros((self.nParticles, self.DoF))
            zeros[n] = copy.deepcopy(-1 * self.gkx[m, n]) # Negate because derivative on second slot.
            zeros = zeros.flatten()
            block = np.einsum('ij, z -> ijz', np.eye(self.DoF), zeros)
            res[m * self.DoF : self.DoF * (m + 1), n * self.DoF : self.DoF * (n + 1), :] = block
        res /= self.nParticles
        rep_a = np.mean(self.gkx, axis=0).flatten()
        rep_b = np.einsum('acc -> a', res)
        assert np.allclose(rep_a, rep_b, rtol=self.rtol, atol=self.atol)
        return res

    def formK_for_numerical_derivative(self, X):
        # Input is an m*b vector
        X = X.reshape(self.nParticles, self.DoF)
        deltas = self.stein.getDeltas(X, X)
        metricDeltas = self.stein.getMetricDeltas(self.M, deltas)
        deltaMetricDeltas = self.stein.getDeltasMetricDeltas(deltas, metricDeltas)
        kx = self.stein.getKernelPiecewise(self.bandwidth, deltaMetricDeltas)
        res = np.zeros((self.dim, self.dim))
        for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
            block = kx[m, n] * np.eye(self.DoF)
            res[m * self.DoF : self.DoF * (m + 1), n * self.DoF : self.DoF * (n + 1)] = block
        return res / self.nParticles

    def getBlock(self, mat, dim, m, n):
        # Works for both N x N blocks and D x D blocks.
        block = mat[m * dim : dim * (m + 1), n * dim : dim * (n + 1)]
        return block

    def makeFullMatrixFromBlockDiagonal(self, mbd):
        res = np.zeros((self.dim, self.dim))
        for m in range(self.nParticles):
            res[m * self.DoF : self.DoF * (m + 1), m * self.DoF : self.DoF * (m + 1)] = mbd[m]
        return res


    # (2) ///////////////////////////
    def test_reshapeNDNDtoNNDD(self):
        mat_NDND = np.random.rand(self.dim, self.dim)
        mat_NNDD = self.stein.reshapeNDNDtoNNDD(mat_NDND)
        for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
            block = self.getBlock(mat_NDND, self.DoF, m, n)
            assert(np.allclose(block, mat_NNDD[m, n], rtol=self.rtol, atol=self.atol))

    # (3) ///////////////////////////
    def test_reshapeNDNDtoDDNN(self):
        mat_NDND = np.random.rand(self.dim, self.dim)
        mat_DDNN = self.stein.reshapeNDNDtoDDNN(mat_NDND)
        for m, n in itertools.product(range(self.DoF), range(self.DoF)):
            block = self.getBlock(mat_NDND, self.nParticles, m, n)
            assert(np.allclose(block, mat_DDNN[m, n], rtol=self.rtol, atol=self.atol))


    ##########################
    # Stochastic SVGD methods
    ##########################
    def form_D_K(self):
        D_K = np.zeros((self.dim, self.dim))
        for d in range(self.DoF):
            D_K[d * self.nParticles : self.nParticles * (d + 1), d * self.nParticles : self.nParticles * (d + 1)] = self.kx
        return D_K / self.nParticles

    ######################################
    # Testing BD SVN-Hessian
    ######################################
    def noise_sampling(self):
        nsamples = 1000
        # B = np.random.multivariate_normal(np.zeros(self.dim), self.HBD_inv_ndnd).reshape(self.nParticles, self.DoF)
        samples = np.random.multivariate_normal(np.zeros(self.dim), self.HBD_inv_ndnd, nsamples)
        test = samples.reshape(nsamples * self.nParticles, self.DoF)
        x = test[:, 0]
        y = test[:, 1]
        for m in range(nsamples):
            plt.scatter(x[m * self.nParticles : self.nParticles * (m + 1)], y[m * self.nParticles : self.nParticles * (m + 1)])
        plt.show()
        pass


    def form_D_SVN_for_numerical_derivative(self, X):
        X = X.reshape(self.nParticles, self.DoF)
        deltas = self.stein.getDeltas(X, X)
        metricDeltas = self.stein.getMetricDeltas(self.M, deltas)
        deltaMetricDeltas = self.stein.getDeltasMetricDeltas(deltas, metricDeltas)
        kx = self.stein.getKernelPiecewise(self.bandwidth, deltaMetricDeltas)

        K = np.zeros((self.dim, self.dim))
        for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
            block = kx[m, n] * np.eye(self.DoF)
            K[m * self.DoF : self.DoF * (m + 1), n * self.DoF : self.DoF * (n + 1)] = block
        K = K / self.nParticles

        gkx = self.stein.getGradKernelPiecewise(self.bandwidth, kx, metricDeltas)
        GN_Hmlpt = self.stein.getGNHessianMinusLogPosterior_ensemble_new(X)
        h_ij_BD = self.stein.h_ij_BD(GN_Hmlpt, kx, gkx)
        HBD_inv = tf.linalg.inv(h_ij_BD)
        HBD_inv_ndnd = self.makeFullMatrixFromBlockDiagonal(HBD_inv)

        return self.nParticles * K @ HBD_inv_ndnd @ K


        # test_a = GK - bracket
        # test_b = GK_test - bracket_test
        # Timing tests
        # import functools
        # import timeit
        # t_contract = timeit.Timer(functools.partial(contract_ijemo, tmp2, self.A_BD_nndd, self.A_BD_nndd))
        # t_contract = timeit.Timer(functools.partial(contract_ijemo, tmp2, self.A_BD_nndd, self.A_BD_nndd))

        # test_b = self.nParticles * (np.einsum('abe, be -> a', self.gradK, self.A_BD_ndnd) -
        #                             np.einsum('aA, ABe, Be -> a', self.A_BD_ndnd.T, self.gradHBD_numerical, self.A_BD_ndnd)).reshape(self.nParticles, self.DoF)

        # assert np.allclose(test_a, test_b, rtol=1e-6, atol=1e-6)

        # Another way to express test_b
        # test_c = self.nParticles * (np.einsum('abe, bc, ce -> a', self.gradK, self.HBD_inv_ndnd, self.K) -
        #                             np.einsum('ab, bA, ABe, Bc, ce -> a', self.K, self.HBD_inv_ndnd, self.gradHBD_numerical, self.HBD_inv_ndnd, self.K)).reshape(self.nParticles, self.DoF)


        # def test_determinstic_correction_math(self):
    #     R = np.einsum('oije, om, on -> omnije', self.grad_GN_Hmlpt, self.kx, self.kx)
    #     B = np.einsum('oij, ome, on -> onmije', self.GN_Hmlpt, self.gkx, self.kx)
    #     C = np.einsum('onie, omj -> omnjie', self.Hesskx, self.gkx)
    #
    #     tmp1 = B + C
    #     tmp2 = tmp1 + R
    #
    #     bracket = np.einsum('onmije, mpif, noje -> pf', tmp2, self.A_nndd, self.A_nndd) \
    #             + np.einsum('omnije, mpif, noje -> pf', B, self.A_nndd, self.A_nndd) \
    #             + np.einsum('omnjie, mpif, noje -> pf', C, self.A_nndd, self.A_nndd) \
    #             - np.einsum('onmije, mpif, nmje -> pf', tmp1, self.A_nndd, self.A_nndd) \
    #             - np.einsum('omnije, mpif, nnje -> pf', B, self.A_nndd, self.A_nndd) \
    #             - np.einsum('omnjie, mpif, nnje -> pf', C, self.A_nndd, self.A_nndd) \
    #             + np.einsum('mne, nmie -> mi', self.gkx, self.A_nndd) \
    #             - np.einsum('mne, nnie -> mi', self.gkx, self.A_nndd)
    #
    #     test_b = self.stein.getDeterministicCorrection_simplemath(self.kx, self.gkx, self.Hesskx, self.GN_Hmlpt, self.grad_GN_Hmlpt, self.A_nndd)
    #     assert np.allclose(test_b, bracket, rtol=1e-6, atol=1e-6)

    # def test_deterministic_correction(self):
    #     # Testing \nabla K : A
    #     test_a = np.einsum('mne, nmie -> mi', self.gkx, self.A_nndd) - np.einsum('mne, nnie -> mi', self.gkx, self.A_nndd)
    #     test_b = self.nParticles * np.einsum('abc, bc -> a', self.gradK, self.A_ndnd).reshape(self.nParticles, self.DoF)
    #     assert np.allclose(test_a, test_b, rtol=1e-6, atol=1e-6)




    # def test_gradK_numerically(self):
    #     # Test that gradK is numerically correct, and that it produces the correct divergence term.
    #     X_flattened = self.X.flatten()
    #     grad_numerical = nd.Gradient(self.formK_for_numerical_derivative)(X_flattened)
    #     grad_numerical = np.einsum('ijk -> kij', grad_numerical) # Recall that numdifftools puts derivative index first
    #     assert np.allclose(self.gradK, grad_numerical, rtol=1e-6, atol=1e-6)
    #     rep_a = np.mean(self.gkx, axis=0).flatten()
    #     rep_b = np.einsum('acc -> a', grad_numerical)
    #     assert np.allclose(rep_a, rep_b, rtol=1e-6, atol=1e-6)

# def test_deterministic_correction_a(self):
#     test_a = np.einsum('mne, nmie -> mi', self.gkx, self.A_nndd) \
#             - np.einsum('mne, nnie -> mi', self.gkx, self.A_nndd)
#     test_b = self.nParticles * np.einsum('abc, bc -> a', self.gradK, self.A_ndnd).reshape(self.nParticles, self.DoF)
#     assert np.allclose(test_a, test_b, rtol=1e-6, atol=1e-6)
#     test_c = self.stein.getDeterministicCorrection_a(self.gkx, self.A_nndd)
#     assert np.allclose(test_a, test_c, rtol=1e-6, atol=1e-6)

# def test_deterministic_correction_b(self):
#     # this guy is kept the same!
#     a_tmp = np.einsum('oije, om, on -> mnoije', self.grad_GN_Hmlpt, self.kx, self.kx) \
#         + np.einsum('oij, ome, on -> mnoije', self.GN_Hmlpt, self.gkx, self.kx) \
#         + np.einsum('oij, om, one -> mnoije', self.GN_Hmlpt, self.kx, self.gkx) \
#         + np.einsum('onie, omj -> mnoije', self.Hesskx, self.gkx) \
#         + np.einsum('oni, omje -> mnoije', self.gkx, self.Hesskx)
#
#     a = np.einsum('mnoije, mpif, noje -> pf', a_tmp, self.A_nndd, self.A_nndd)
#
#     # This guy requires a bit of modification
#     b_tmp = np.einsum('Nij, Noe, Nn -> noije', -1 * self.GN_Hmlpt, self.gkx, self.kx) \
#         - np.einsum('Nni, Noje -> noije', self.gkx, self.Hesskx)
#     b = np.einsum('noije, opif, noje -> pf', b_tmp, self.A_nndd, self.A_nndd)
#
#     c_tmp = np.einsum('Nij, Nm, Nne -> mnije', -1 * self.GN_Hmlpt, self.kx, self.gkx) \
#         - np.einsum('Nnie, Nmj -> mnije', self.Hesskx, self.gkx)
#     c = np.einsum('mnije, mpif, nnje -> pf', c_tmp, self.A_nndd, self.A_nndd)
#
#     test_a = a + b + c
#
#     test_b = self.nParticles * np.einsum('ab, bce, ce -> a', self.A_ndnd.T, self.gradH_numerical, self.A_ndnd).reshape(self.nParticles, self.DoF)
#     assert np.allclose(test_a, test_b, rtol=1e-6, atol=1e-6)
#     test_c = self.stein.getDeterministicCorrection_b(self.kx, self.gkx, self.Hesskx, self.GN_Hmlpt, self.grad_GN_Hmlpt, self.A_nndd)
#     assert np.allclose(test_a, test_c, rtol=1e-6, atol=1e-6)
#########################################################################
# Stochastic SVN: Action methods (5)
#########################################################################

##################################################################
# Stochastic SVN Block Diagonal: Matrix-vector product methods (2)
##################################################################

# def test_deterministic_correction_numerically(self):
#     # Check if deterministic correction is correct by confirming the following:
#     # D^{SVN} \nabla \ln \pi + \nabla \cdot D^{SVN} = v^{SVN} + v^{cor}
#     X_flattened = self.X.flatten()
#     grad_numerical = nd.Gradient(self.form_D_SVN_for_numerical_derivative)(X_flattened)
#     grad_numerical = np.einsum('ijk -> kij', grad_numerical)
#     div_D_SVN = np.einsum('abb -> a', grad_numerical).reshape(self.nParticles, self.DoF)
#     uphill = (self.D_SVN @ (-1 * self.gmlpt).flatten()).reshape(self.nParticles, self.DoF)
#     a = uphill + div_D_SVN
#     v_svn = self.stein.getSVN_Direction(self.kx, self.gkx, self.gmlpt, self.K_HBD_inv, self.HBD_inv_K)
#     v_deterministic_correction = self.stein.getSVN_deterministic_correction(self.gkx, self.grad_hij_BD, self.HBD_inv_K, self.K_HBD_inv)
#     b = v_svn + v_deterministic_correction
#     assert(np.allclose(a, b, rtol=1e-8, atol=1e-8))


# C_ije = np.einsum('omjie -> omije', C)

# assert np.allclose(R, R_test2, rtol=1e-6, atol=1e-6)
# assert np.allclose(B, B_test2, rtol=1e-6, atol=1e-6)
# assert np.allclose(C, C_test2, rtol=1e-6, atol=1e-6)
# Test: H^{-1} K
# test1 = self.HBD_inv_ndnd @ self.K
# assert np.allclose(self.A_BD_ndnd, test1, rtol=1e-6, atol=1e-6)

# Test: Block values of R, B, C
# R_test1 = np.einsum('oije, om, on -> omnije', self.grad_GN_Hmlpt, self.kx, self.kx)
# B_test1 = np.einsum('oij, ome, on -> onmije', self.GN_Hmlpt, self.gkx, self.kx)
# C_test1 = np.einsum('onie, omj -> omnjie', self.Hesskx, self.gkx)

# R_test2 = R_test1[:, range(self.nParticles), range(self.nParticles)]
# B_test2 = B_test1[:, range(self.nParticles), range(self.nParticles)]
# C_test2 = C_test1[:, range(self.nParticles), range(self.nParticles)]

# Calculate temp variables
# bracket = np.einsum('omije, mpif, moje -> pf', tmp2, self.A_BD_nndd, self.A_BD_nndd) \
#         - np.einsum('omije, mpif, mmje -> pf', tmp1, self.A_BD_nndd, self.A_BD_nndd) \
#         + np.einsum('omjie, mpif, moje -> pf', C, self.A_BD_nndd, self.A_BD_nndd) \
#         - np.einsum('omjie, mpif, mmje -> pf', C, self.A_BD_nndd, self.A_BD_nndd)

# assert np.allclose(bracket, bracket_check, rtol=1e-6, atol=1e-6)


def main():
    a = Test_sanity()
    a.setup()
    # a.test_deterministic_correction_numerically()
    # a.test_gradK_numerically()
    # a.test_gradH_form()
    # a.test_AT_gradH_A()
    # a.test_stochastic_correction()
    # a.test_NKA()
    # a.test_deterministic_correction()
    # a.test_determinstic_correction_math()
    # a.test_getSVN_direction()
    # a.test_getBD_Deterministic_correction()
    # a.test_gradHBD_form_simpler()
    # a.test_gradHBD_form_simpler()
    # a.test_getStochastic_correction()
    a.test_CG_solver()
if __name__ is '__main__':
    main()