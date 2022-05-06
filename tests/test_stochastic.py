import numpy as np
import itertools
from models.rosenbrock_analytic_proper_sample import rosenbrock_analytic
from source.stein_experiments import SVI
import os
import copy
import scipy
import tensorflow as tf
import numdifftools as nd
import matplotlib.pyplot as plt
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

        ##############################################################
        # Calculations needed to perform unit tests
        ##############################################################
        self.M = np.eye(self.DoF)
        self.deltas = self.stein.getDeltas(self.X, self.X)
        self.metricDeltas = self.stein.getMetricDeltas(self.M, self.deltas)
        self.deltasMetricDeltas = self.stein.getDeltasMetricDeltas(self.deltas, self.metricDeltas)
        self.bandwidth = self.stein.getBandwidth(self.deltasMetricDeltas)
        self.kernel = self.stein.getKernelPiecewise(self.bandwidth, self.deltasMetricDeltas)
        self.gradKernel = self.stein.getGradKernelPiecewise(self.bandwidth, self.kernel, self.metricDeltas)
        self.Hesskx = self.stein.getHessianKernel(self.bandwidth, self.kernel, self.gradKernel, self.M, self.metricDeltas)
        self.gmlpt = self.stein.getGradientMinusLogPosterior_ensemble_new(self.X)
        self.GN_Hmlpt = self.stein.getGNHessianMinusLogPosterior_ensemble_new(self.X)
        self.H_bar = self.stein.H_bar(self.GN_Hmlpt, self.kernel, self.gradKernel)
        ##############################################################
        # New calculations needed to perform unit tests
        ##############################################################
        self.HBD = self.stein.H_bar_BD(self.GN_Hmlpt, self.kernel, self.gradKernel)
        self.HBD_inv = tf.linalg.pinv(self.HBD).numpy()
        self.HBD_inv_sqrt = tf.linalg.cholesky(self.HBD_inv).numpy()
        self.K_HBD_inv = self.stein.K_action_mbd(self.kernel, self.HBD_inv)
        self.HBD_inv_K = np.einsum('mnbd -> nmdb', self.K_HBD_inv)
        self.dim = self.nParticles * self.DoF
        self.grad_GN_Hmlpt = self.stein.getGradientGNHessianMinusLogPosterior_ensemble(self.X)
        self.grad_hij_BD = self.stein.grad_hij_BD_TESTING(self.kernel, self.gradKernel, self.Hesskx, self.GN_Hmlpt, self.grad_GN_Hmlpt)

        self.K = self.formK()
        self.D_K = self.form_D_K()
        self.HBD_inv_ndnd = self.makeFullMatrixFromBlockDiagonal(self.HBD_inv)
        self.D_SVN = self.nParticles * self.K @ self.HBD_inv_ndnd @ self.K
        self.gradK = self.form_gradK_first_slot() + self.form_gradK_second_slot()
        self.gradH = self.form_grad_hij_BD()
        self.divK = np.einsum('abb -> a', self.gradK)

        # LAYOUT OF UNIT TESTS
        # (i) Helper methods:
        # Form \nabla_z K and \nabla_z H manually (also contains auxilliary methods).

        # (ii) Reshaping methods:
        # Necessary for common tensor manipulations.

        # (iii) Action methods:
        # Reduces memory requirements for calculations over the latent space by accounting for sparsity.

        # (iv) Mat-vecs:
        # Contractions that take into account block structure of matricies.

        # (v) Math:
        # Consistency checks for all calculations.

    ############################################################
    # Stochastic SVN: Helper methods
    ############################################################

    def form_h_ij_BD_for_numerical_derivative(self, X):
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
            block = self.kernel[m, n] * np.eye(self.DoF)
            res[m * self.DoF : self.DoF * (m + 1), n * self.DoF : self.DoF * (n + 1)] = block
        return res / self.nParticles

    def form_gradK_first_slot(self):
        # Tensor contribution to gradK from first kernel slot
        dim = self.nParticles * self.DoF
        res = np.zeros((dim, dim, dim))
        # Get the gradient of augmented kernel matrix
        for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
            zeros = np.zeros((self.nParticles, self.DoF))
            zeros[m] = copy.deepcopy(self.gradKernel[m, n])
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
            zeros[n] = copy.deepcopy(-1 * self.gradKernel[m, n]) # Negate because derivative on second slot.
            zeros = zeros.flatten()
            block = np.einsum('ij, z -> ijz', np.eye(self.DoF), zeros)
            res[m * self.DoF : self.DoF * (m + 1), n * self.DoF : self.DoF * (n + 1), :] = block
        res /= self.nParticles
        rep_a = np.mean(self.gradKernel, axis=0).flatten()
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

    def test_gradK_numerically(self):
        # Test that gradK is numerically correct, and that it produces the correct divergence term.
        X_flattened = self.X.flatten()
        grad_numerical = nd.Gradient(self.formK_for_numerical_derivative)(X_flattened)
        grad_numerical = np.einsum('ijk -> kij', grad_numerical) # Recall that numdifftools puts derivative index first
        assert np.allclose(self.gradK, grad_numerical, rtol=1e-6, atol=1e-6)
        rep_a = np.mean(self.gradKernel, axis=0).flatten()
        rep_b = np.einsum('acc -> a', grad_numerical)
        assert np.allclose(rep_a, rep_b, rtol=1e-6, atol=1e-6)

    def getBlock(self, mat, dim, m, n):
        # Works for both N x N blocks and D x D blocks.
        block = mat[m * dim : dim * (m + 1), n * dim : dim * (n + 1)]
        return block

    def makeFullMatrixFromBlockDiagonal(self, mbd):
        res = np.zeros((self.dim, self.dim))
        for m in range(self.nParticles):
            res[m * self.DoF : self.DoF * (m + 1), m * self.DoF : self.DoF * (m + 1)] = mbd[m]
        return res

    #########################################################################
    # Stochastic SVN: Reshaping methods (3)
    #########################################################################

    # (1) ///////////////////////////
    def test_reshapeNNDDtoNDND(self):
        mat_NNDD = np.random.rand(self.nParticles, self.nParticles, self.DoF, self.DoF)
        mat_NDND = self.stein.reshapeNNDDtoNDND(mat_NNDD)
        for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
            block = self.getBlock(mat_NDND, self.DoF, m, n)
            assert(np.allclose(block, mat_NNDD[m, n], rtol=self.rtol, atol=self.atol))

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

    #########################################################################
    # Stochastic SVN: Action methods (5)
    #########################################################################

    # (1) /////////////////////
    def test_K_action_mb(self):
        # Compare (minus) uphill SVGD direction contribution
        test_a = self.stein.K_action_mb(self.kernel, self.gmlpt)
        test_b = self.stein.contract_term_mgJ(self.kernel, self.gmlpt) / self.nParticles
        assert(np.allclose(test_a, test_b, rtol=self.rtol, atol=self.atol))

        # Compare direct action
        B = np.random.rand(self.nParticles, self.DoF)
        test_two = self.stein.K_action_mb(self.kernel, B)
        test_one = (self.K @ B.flatten()).reshape(self.nParticles, self.DoF)
        assert(np.allclose(test_one, test_two, rtol=self.rtol, atol=self.atol))

    # (2) //////////////////////
    def test_K_action_mbd(self):
        # Form K and directly compare action mbd matrix
        K = self.formK()
        Bndd = np.random.rand(self.nParticles, self.DoF, self.DoF)
        test_a = self.stein.K_action_mbd(self.kernel, Bndd)
        Bndnd = self.makeFullMatrixFromBlockDiagonal(Bndd)
        test_b = self.stein.reshapeNDNDtoNNDD(K @ Bndnd)
        assert(np.allclose(test_a, test_b, rtol=self.rtol, atol=self.atol))

    # (3) ///////////////////////
    def test_K_action_mnbd(self):
        # Form K and directly compare action mnbd matrix
        K = self.formK()
        Bnndd = np.random.rand(self.nParticles, self.nParticles, self.DoF, self.DoF)
        test_a = self.stein.K_action_mnbd(self.kernel, Bnndd)
        Bndnd = self.stein.reshapeNNDDtoNDND(Bnndd)
        test_b = self.stein.reshapeNDNDtoNNDD(K @ Bndnd)
        assert(np.allclose(test_a, test_b, rtol=self.rtol, atol=self.atol))

    # (4) ///////////////////////////
    def test_gradK_action_mnbd(self):
        # Get action by forming individual gradient contributions: \nabla_1 K A + \nabla_2 K A
        Bnndd = np.random.rand(self.nParticles, self.nParticles, self.DoF, self.DoF)
        test_a = self.stein.gradK_action_mnbd(self.gradKernel, Bnndd)
        Bndnd = self.stein.reshapeNNDDtoNDND(Bnndd)
        res_a = self.form_gradK_first_slot()
        res_b = self.form_gradK_second_slot()
        vec_naive_a = np.einsum('abc, bc -> a', res_a, Bndnd) # First slot contribution (None)
        vec_naive_b = np.einsum('abc, bc -> a', res_b, Bndnd) # Second slot contribution
        test_b = (vec_naive_a + vec_naive_b).reshape(self.nParticles, self.DoF)
        assert np.allclose(test_a, test_b, rtol=self.rtol, atol=self.atol)

    # (5) /////////////////////////////////
    def test_grad_hij_BD_action_mnbd(self):
        Bnndd = np.random.rand(self.nParticles, self.nParticles, self.DoF, self.DoF)
        Bndnd = self.stein.reshapeNNDDtoNDND(Bnndd)
        test_a = self.stein.grad_hij_BD_action_mnbd(self.grad_hij_BD, Bnndd).flatten()
        grad_hij_BD_matrix = self.form_grad_hij_BD()
        test_b = np.einsum('abc, bc -> a', grad_hij_BD_matrix, Bndnd)
        assert np.allclose(test_a, test_b, rtol=1e-8, atol=1e-8)

    ##################################################################
    # Stochastic SVN Block Diagonal: Matrix-vector product methods (2)
    ##################################################################

    # (1) ////////////////////////
    def test_mnbd_mb_matvec(self):
        mat = np.random.rand(self.nParticles, self.nParticles, self.DoF, self.DoF)
        vec = np.random.rand(self.nParticles, self.DoF)
        test_a = self.stein.mnbd_mb_matvec(mat, vec)
        test_b = (self.stein.reshapeNNDDtoNDND(mat) @ vec.flatten()).reshape(self.nParticles, self.DoF)
        assert np.allclose(test_a, test_b, rtol=self.rtol, atol=self.atol)

    # (2) ///////////////////////
    def test_mbd_mb_matvec(self):
        mnb = np.random.rand(self.nParticles, self.DoF, self.DoF)
        mat = self.makeFullMatrixFromBlockDiagonal(mnb)
        vec = np.random.rand(self.nParticles, self.DoF)
        test_a = self.stein.mbd_mb_matvec(mnb, vec)
        test_b = (mat @ vec.flatten()).reshape(self.nParticles, self.DoF)
        assert np.allclose(test_a, test_b, rtol=self.rtol, atol=self.atol)

    ############################################################
    # Stochastic SVN: Math
    ############################################################

    # (0) /////////////////
    def test_h_ij_BD(self):
        # Makes sure block diagonal calculation agrees with full Hessian calculation.
        test_a = self.stein.h_ij_BD(self.GN_Hmlpt, self.kernel, self.gradKernel)
        H_bar_nndd = self.stein.reshapeNDNDtoNNDD(self.H_bar)
        for m in range(self.nParticles):
            assert(np.allclose(test_a[m], H_bar_nndd[m, m], rtol=self.rtol, atol=self.atol))

    # (1) ////////////////////////////////////
    def test_gradient_hij_BD_numerically(self):
        # Form \nabla_z H manually and compare to numerical Jacobian
        grad_numerical = nd.Gradient(self.form_h_ij_BD_for_numerical_derivative)(self.X.flatten())
        grad_numerical = np.einsum('ijk -> kij', grad_numerical)
        assert np.allclose(grad_numerical, self.gradH, rtol=1e-8, atol=1e-8)

    # (2) //////////////////////////
    def test_getSVN_Direction(self):
        # Check that the SVN direction matches with the direct calculation
        test_a = self.stein.getSVN_Direction(self.kernel, self.gradKernel, self.gmlpt, self.K_HBD_inv, self.HBD_inv_K)
        D_SVN_test = self.nParticles * self.K @ self.HBD_inv_ndnd @ self.K
        a = D_SVN_test @ (-1 * self.gmlpt).flatten()
        b = self.nParticles * self.K @ self.HBD_inv_ndnd @ self.divK
        test_b = (a + b).reshape(self.nParticles, self.DoF)
        assert(np.allclose(test_a, test_b, rtol=1e-8, atol=1e-8))

    # (3) //////////////////////////////////
    def test_deterministic_correction_analytically(self):
        test_a = self.stein.getSVN_deterministic_correction(self.gradKernel, self.grad_hij_BD, self.HBD_inv_K, self.K_HBD_inv)
        test_b = self.nParticles * (np.einsum('abe, bc, ce -> a', self.gradK, self.HBD_inv_ndnd, self.K)
                                    - np.einsum('ab, bA, ABe, Bc, ce -> a', self.K, self.HBD_inv_ndnd, self.gradH, self.HBD_inv_ndnd, self.K)).reshape(self.nParticles, self.DoF)
        assert(np.allclose(test_a, test_b, rtol=1e-8, atol=1e-8))

    def test_deterministic_correction_numerically(self):
        # Check if deterministic correction is correct by confirming the following:
        # D^{SVN} \nabla \ln \pi + \nabla \cdot D^{SVN} = v^{SVN} + v^{cor}
        X_flattened = self.X.flatten()
        grad_numerical = nd.Gradient(self.form_D_SVN_for_numerical_derivative)(X_flattened)
        grad_numerical = np.einsum('ijk -> kij', grad_numerical)
        div_D_SVN = np.einsum('abb -> a', grad_numerical).reshape(self.nParticles, self.DoF)
        uphill = (self.D_SVN @ (-1 * self.gmlpt).flatten()).reshape(self.nParticles, self.DoF)
        a = uphill + div_D_SVN
        v_svn = self.stein.getSVN_Direction(self.kernel, self.gradKernel, self.gmlpt, self.K_HBD_inv, self.HBD_inv_K)
        v_deterministic_correction = self.stein.getSVN_deterministic_correction(self.gradKernel, self.grad_hij_BD, self.HBD_inv_K, self.K_HBD_inv)
        b = v_svn + v_deterministic_correction
        assert(np.allclose(a, b, rtol=1e-8, atol=1e-8))

    def test_SVN_stochastic_correction(self):
        # Calculate manually and compare to method
        B = np.random.normal(0, 1, (self.nParticles, self.DoF))
        test_a = self.stein.getSVN_stochastic_correction(self.kernel, self.HBD_inv_sqrt, B=B)
        L_HBD_inv = tf.linalg.cholesky(self.HBD_inv_ndnd).numpy()
        test_b = np.sqrt(2 * self.nParticles) * (self.K @ L_HBD_inv @ B.flatten()).reshape(self.nParticles, self.DoF)
        assert np.allclose(test_a, test_b, rtol=1e-6, atol=1e-6)

        # Confirm that the samples drawn agree with the naive implementation
        # p_a = np.random.multivariate_normal(np.zeros(self.dim), self.HBD_inv_ndnd, 200)
        # p_b = self.K @ L_HBD_inv @ B.flatten()


        # for m in range(self.nParticles):
        #     block = self.getBlock(L_H_inv, self.DoF, m, m)
        #     assert np.allclose(block, self.HBD_inv_sqrt[m], rtol=1e-6, atol=1e-6)
        # bnd = B.reshape(self.nParticles, self.DoF)
        # Plot samples to make sure they are distributed identically



        # test_b = self.stein.mbd_mb_matvec(self.HBD_inv_sqrt, bnd)
        # test_b = np.einsum('mbd, md -> mb', self.HBD_inv_sqrt, bnd)
        # for m in range(self.nParticles):
        #     assert np.allclose(test_a[m], test_b[m], rtol=1e-6, atol=1e-6)
        #
        # Check that kernel averaging produces expected results
        # test_one = self.stein.K_action_mb(self.kernel, test_a)
        # test_two = (self.K @ test_a.flatten()).reshape(self.nParticles, self.DoF)
        # assert np.allclose(test_one, test_two, rtol=1e-6, atol=1e-6)



    #################################################################
    # Stochastic SVN methods: Testing new methods
    #################################################################
    def form_h_ij_BD_for_numerical_derivative_extra_variables(self, X, bandwidth, metric):
        # Input is an m*b vector
        X = X.reshape(self.nParticles, self.DoF)
        deltas = self.stein.getDeltas(X, X)
        metricDeltas = self.stein.getMetricDeltas(metric, deltas)
        deltaMetricDeltas = self.stein.getDeltasMetricDeltas(deltas, metricDeltas)
        kx = self.stein.getKernelPiecewise(bandwidth, deltaMetricDeltas)
        gkx = self.stein.getGradKernelPiecewise(bandwidth, kx, metricDeltas)
        GN_Hmlpt = self.stein.getGNHessianMinusLogPosterior_ensemble_new(X)
        h_ij_BD = self.stein.h_ij_BD(GN_Hmlpt, kx, gkx)
        H_BD = self.makeFullMatrixFromBlockDiagonal(h_ij_BD)
        return H_BD

    def test_aux_variables_in_numerical_derivative(self):
        X_flattened = self.X.flatten()
        grad_numerical_a = nd.Gradient(self.form_h_ij_BD_for_numerical_derivative)(X_flattened)
        grad_numerical_b = nd.Gradient(self.form_h_ij_BD_for_numerical_derivative_extra_variables)(X_flattened, self.bandwidth, self.M)
        assert np.allclose(grad_numerical_a, grad_numerical_b, rtol=1e-8, atol=1e-8)

    def test_difference_between_jacobian_grad(self):
        X_flattened = self.X.flatten()
        grad_numerical = nd.Gradient(self.form_h_ij_BD_for_numerical_derivative)(X_flattened)
        jac_numerical = nd.Jacobian(self.form_h_ij_BD_for_numerical_derivative)(X_flattened)
        assert np.allclose(jac_numerical, grad_numerical, rtol=1e-8, atol=1e-8)

    def test_permutation_orders(self):
        dim = self.nParticles * self.DoF
        v = np.random.rand(self.nParticles, self.DoF)
        vc = v.reshape(dim, order='C')
        vf = v.reshape(dim, order='F')
        def c_to_f(v):
            # row major to column major
            # C to F
            return (v.reshape(self.nParticles, self.DoF)).reshape(dim, order='F')
        def f_to_c(v):
            # column major to row major
            # F to C
            return (v.reshape(self.DoF, self.nParticles)).reshape(dim, order='F')
        assert np.allclose(vc, f_to_c(vf), rtol=1e-6, atol=1e-6)
        assert np.allclose(vf, c_to_f(vc), rtol=1e-6, atol=1e-6)

        # We now wish to show that the action is identical
        vec = np.random.rand(dim)
        K = self.formK()
        D_K = scipy.linalg.block_diag(self.kernel, self.kernel) / self.nParticles
        test_a = K @ vec
        test_b = f_to_c(D_K @ c_to_f(vec))
        assert np.allclose(test_a, test_b, rtol=1e-6, atol=1e-6)

        # We form an analogous method for matricies and check it works columnwise as expected
        A = np.random.rand(dim, dim)
        mat_c_to_f = lambda A: np.apply_along_axis(c_to_f, 0, A)
        mat_f_to_c = lambda A: np.apply_along_axis(f_to_c, 0, A)
        TEST_a = c_to_f(A[:,0])
        TEST_b = mat_c_to_f(A)[:, 0]
        assert np.allclose(TEST_a, TEST_b, rtol=1e-6, atol=1e-6)

        # Test to see the mat functions invert each other properly
        assert np.allclose(A, mat_c_to_f(mat_f_to_c(A)), rtol=1e-6, atol=1e-6)

        # Form gkx block
        gkx_block = np.zeros((self.nParticles, self.nParticles, dim))
        for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
            zeros = np.zeros((self.nParticles, self.DoF))
            zeros[m] = copy.deepcopy(self.gradKernel[m, n])
            zeros[n] = copy.deepcopy(-1 * self.gradKernel[m, n])
            zeros = zeros.flatten()
            gkx_block[m, n, :] = zeros

        # Form gradD_K naively
        gradD_K = np.zeros((dim, dim, dim))
        for m in range(self.DoF):
            gradD_K[m * self.nParticles : self.nParticles * (m + 1), m * self.nParticles : self.nParticles * (m + 1), :] = gkx_block
        gradD_K /= self.nParticles

        # Directly compare
        # gradK = self.gradK_single_parameter_def()
        testing_a = f_to_c(np.einsum('abc, bc -> a', gradD_K, mat_c_to_f(A)))
        testing_b = np.einsum('abc, bc -> a', self.gradK, A)
        assert np.allclose(testing_a, testing_b, rtol=1e-6, atol=1e-6)
        pass

        # Obtain action in vectorized fashion
        # 1) First entry.
        A_DDNN = self.stein.reshapeNDNDtoDDNN(mat_c_to_f(A))
        gradD_K_block_single = self.stein.reshapeNDNDtoDDNN(gkx_block[0]) / self.nParticles
        t_1 = np.trace(gradD_K_block_single[0,0] @ A_DDNN[0,0].T + gradD_K_block_single[0,1] @ A_DDNN[0,1].T)
        assert np.allclose(t_1, testing_a[0], rtol=1e-6, atol=1e-6)
        # 2) Rewrite using einsum.
        tmp_a = np.einsum('bdmn, dmn -> ', gradD_K_block_single, A_DDNN[0])
        assert np.allclose(tmp_a, testing_b[0], rtol=1e-6, atol=1e-6)
        # 3) Get the first block.
        tmp_b = np.einsum('bdmn, qdmn -> q', gradD_K_block_single, A_DDNN)
        assert np.allclose(tmp_b, testing_b[0:self.DoF], rtol=1e-6, atol=1e-6)
        # 4) Get all of the blocks.
        gradD_K_block = (gkx_block / self.nParticles).reshape(self.nParticles, self.nParticles, 1, self.DoF, self.nParticles).swapaxes(1,3).reshape(self.nParticles, 1, self.DoF, self.nParticles, self.nParticles)
        tmp_c = np.einsum('obdmn, qdmn -> oq', gradD_K_block, A_DDNN).flatten()
        assert np.allclose(tmp_c, testing_b, rtol=1e-6, atol=1e-6)
        # I.E, we C -> F the matrix we would like to get the gradKernel action of, then apply this procedure...
        # Note: it appears as though we've implicitly handled the final C -> F mapping!
        pass

    ##########################
    # Stochastic SVGD methods
    ##########################
    def form_D_K(self):
        D_K = np.zeros((self.dim, self.dim))
        for d in range(self.DoF):
            D_K[d * self.nParticles : self.nParticles * (d + 1), d * self.nParticles : self.nParticles * (d + 1)] = self.kernel
        return D_K / self.nParticles

    def test_SVGD_noise(self):
        # Confirm that sqrt and Cholesky factorizations of BD-matrix is the sqrt of the blocks.
        cholesky_kxon = tf.linalg.cholesky(self.kernel / self.nParticles).numpy()
        cholesky_D_K = tf.linalg.cholesky(self.D_K).numpy()
        sqrt_kx = tf.linalg.sqrtm(self.kernel).numpy()
        sqrt_D_K = tf.linalg.sqrtm(self.D_K).numpy()
        for d in range(self.DoF):
            block = self.getBlock(cholesky_D_K, self.nParticles, d, d)
            assert np.allclose(cholesky_kxon, block, rtol=1e-6, atol=1e-6)
        for d in range(self.DoF):
            block = self.getBlock(sqrt_D_K, self.nParticles, d, d)
            assert np.allclose(sqrt_kx, block, rtol=1e-6, atol=1e-6)
        B = np.random.normal(0, 1, self.dim)
        test_a = (cholesky_D_K @ B).reshape(self.nParticles, self.DoF)
        B_bm = B.reshape(self.DoF, self.nParticles)
        test_b = np.einsum('mn, dn -> dm', cholesky_kxon, B_bm).reshape(self.nParticles, self.DoF, order='C')
        assert np.allclose(test_a, test_b, rtol=1e-6, atol=1e-6)


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




        # test_b = self.nParticles * (np.einsum('ab, bc, c -> a', self.K, self.HBD_inv_ndnd, self.divK)
        #                             + np.einsum('abe, bc, ce -> a', self.gradK, self.HBD_inv_ndnd, self.K)
        #                             - np.einsum('ab, bA, ABe, Bc, ce -> a', self.K, self.HBD_inv_ndnd, self.gradH, self.HBD_inv_ndnd, self.K)).reshape(self.nParticles, self.DoF)
        #
        #
        # test_c = self.stein.getSVN_deterministic_correction(self.gradKernel, self.grad_hij_BD, self.HBD_inv_K, self.K_HBD_inv) \
        #          + self.nParticles * np.einsum('ab, bc, c -> a', self.K, self.HBD_inv_ndnd, self.divK).reshape(self.nParticles, self.DoF)
        # assert(np.allclose(test_a, test_b, rtol=1e-8, atol=1e-8))
        # pass

def main():
    a = Test_sanity()
    a.setup()
    # a.form_gradK_first_slot()
    # a.test_divD()
    # a.test_K_action_mnbd()
    # a.test_getSVN_Direction()
    # a.test_numerical_gradient_h_ij_BD()
    # a.test_grad_hij_BD_action_mnbd()
    # a.formGradH_BD_second_slot()
    # a.test_gradHBD_action_second_slot_mnbd()
    # a.test_aux_variables_in_numerical_derivative()
    # a.test_deterministic_correction()
    # a.form_gradK_second_slot()
    # a.test_SVGD_noise()
    # a.test_SVN_noise()
    # a.noise_sampling()
    # a.test_D_SVN_numerically()
    # a.test_deterministic_correction_numerically()
    a.test_gradient_hij_BD_numerically()
    # a.test_stochastic_correction()
if __name__ is '__main__':
    main()