import numpy as np
import scipy
import itertools
from models.rosenbrock_analytic_proper_sample import rosenbrock_analytic
from source.stein_experiments import samplers
import tensorflow as tf
import numdifftools as nd
from tools.kernels import getGaussianKernelWithDerivatives_metric
from opt_einsum import contract
# import matplotlib.pyplot as plt

class Test_sanity:
    def setup(self):
        np.random.seed(1)
        # Setup SVI object
        model = rosenbrock_analytic()
        self.rtol = 1e-15
        self.atol = 1e-15
        self.optimizeMethod = 'SVN'
        self.nParticles = 4
        self.nIterations = 1
        self.stein = samplers(model = model, nParticles=self.nParticles, nIterations=self.nIterations)
        self.X = np.copy(self.stein.model.newDrawFromPrior(self.nParticles))
        self.DoF = self.stein.DoF
        # Define objects that will be convenient in tests
        self.dim = self.nParticles * self.DoF
        self.delta_dim = np.eye(self.dim)
        self.delta_N = np.eye(self.nParticles)
        self.delta_D = np.eye(self.DoF)
        ##############################################################
        # Calculations: Vanilla SVN
        ##############################################################
        self.M = np.eye(self.DoF)
        self.kx, self.gkx, self.Hesskx = getGaussianKernelWithDerivatives_metric(self.X, get_hesskx=True)
        self.gmlpt = self.stein.getGradientMinusLogPosterior_ensemble_new(self.X)
        self.GN_Hmlpt = self.stein.getGNHessianMinusLogPosterior_ensemble_new(self.X)
        self.H_bar = self.stein.H_bar(self.GN_Hmlpt, self.kx, self.gkx)
        #########################################################################
        # Calculations: Form the matricies to be used in comparison
        #########################################################################
        # *** Full/BD agnostic calculations ***
        self.K = self.formK_for_numerical_derivative(self.X)
        self.gradK = np.einsum('ijk -> kij', nd.Gradient(self.formK_for_numerical_derivative)(self.X))
        self.divK = np.einsum('abb -> a', self.gradK)
        self.D_K = self.form_D_K() # SVGD diffusion matrix
        self.grad_GN_Hmlpt = self.stein.getGradientGNHessianMinusLogPosterior_ensemble(self.X)
        # *** BD calculations ***
        self.HBD = self.stein.H_bar_BD(self.GN_Hmlpt, self.kx, self.gkx)
        self.HBD_inv = tf.linalg.inv(self.HBD).numpy()
        self.HBD_inv_ndnd = self.makeFullMatrixFromBlockDiagonal(self.HBD_inv)
        self.D_SVN_BD = self.nParticles * self.K @ self.HBD_inv_ndnd @ self.K # SVN diffusion matrix
        self.L_BD_ndd = tf.linalg.cholesky(self.HBD_inv).numpy()
        self.L_BD_ndnd = self.makeFullMatrixFromBlockDiagonal(self.L_BD_ndd)
        self.A_BD_ndnd = self.HBD_inv_ndnd @ self.K
        self.A_BD_nndd = self.stein.reshapeNDNDtoNNDD(self.A_BD_ndnd)
        self.gradHBD_numerical = np.einsum('ijk -> kij', nd.Gradient(self.form_HBD_for_numdiff)(self.X.flatten()))
        self.grad_D_SVN_BD = np.einsum('ijk -> kij', nd.Gradient(self.form_D_SVN_BD_for_numerical_derivative)(self.X))
        self.div_D_SVN_BD = np.einsum('abb -> a', self.grad_D_SVN_BD)

    def test_getSVGD_Stochastic_correction(self):
        """
        Make sure getSVGD_Stochastic_correction returns naive noise after eq (6)
        """
        B = np.random.normal(0, 1, self.dim)
        Bdn = B.reshape(self.DoF, self.nParticles)
        L_D_K = np.linalg.cholesky(self.D_K)
        L_kx = np.linalg.cholesky(self.kx)
        test_a = self.stein.getSVGD_Stochastic_correction(L_kx, Bdn)
        test_b = np.sqrt(2) * (L_D_K @ B).reshape(self.nParticles, self.DoF)
        assert np.allclose(test_a, test_b, atol=1e-6)

    # (1)
    def test_getSVN_direction(self):
        """
        Check if getSVN_direction returns the first term in eq (20)
        """
        test_a = self.stein.getSVN_direction(self.kx, self.gkx, self.gmlpt, self.A_BD_nndd)
        test_b = self.nParticles * self.K @ self.HBD_inv_ndnd @ (-self.K @ self.gmlpt.flatten() + self.divK)
        assert np.allclose(test_a, test_b.reshape(self.nParticles, self.DoF), rtol=1e-6, atol=1e-6)

    # (2b)
    def test_getSVN_BD_Deterministic_correction(self):
        """
        Check if getSVN_BD_Deterministic_correction returns second term in eq (20)
        """
        test_a = self.stein.getSVN_BD_Deterministic_correction(self.kx, self.gkx, self.Hesskx, self.GN_Hmlpt, self.grad_GN_Hmlpt, self.A_BD_nndd)
        test_b = self.nParticles * np.einsum('abe, be -> a', self.gradK, self.A_BD_ndnd) \
                - self.nParticles * np.einsum('aA, ABe, Be -> a', self.A_BD_ndnd.T, self.gradHBD_numerical, self.A_BD_ndnd)
        assert np.allclose(test_a, test_b.reshape(self.nParticles, self.DoF), rtol=1e-6, atol=1e-6)

    # (3)
    def test_getSVN_BD_Stochastic_correction(self):
        """
        Check if getSVN_BD_Stochastic_correction returns naive noise before eq (29)
        """
        B = np.random.normal(0, 1, (self.nParticles, self.DoF))
        test_a = self.stein.getSVN_BD_Stochastic_correction(self.kx, self.L_BD_ndd, B)
        test_b = np.sqrt(2 * self.nParticles) * (self.K @ self.L_BD_ndnd @ B.flatten()).reshape(self.nParticles, self.DoF)
        assert np.allclose(test_a, test_b, rtol=1e-6, atol=1e-6)

    # (Aux 1)
    def test_sSVN_uphill_numerical_divergence(self):
        """
        Check if numerical and analytical calculation of deterministic part of update agree
        """
        test_a = self.stein.getSVN_direction(self.kx, self.gkx, self.gmlpt, self.A_BD_nndd) \
               + self.stein.getSVN_BD_Deterministic_correction(self.kx, self.gkx, self.Hesskx, self.GN_Hmlpt, self.grad_GN_Hmlpt, self.A_BD_nndd)
        test_b = - self.D_SVN_BD @ self.gmlpt.flatten() + self.div_D_SVN_BD
        assert np.allclose(test_a, test_b.reshape(self.nParticles, self.DoF), rtol=1e-6, atol=1e-6)

    # (Aux 2)
    def test_getA_BD(self):
        """
        Confirm that getA_BD is calculated correctly
        """
        test_a = self.stein.getA_BD(self.HBD_inv, self.kx)
        assert np.allclose(test_a, self.A_BD_nndd, atol=1e-6)

    ################################################################
    # Stochastic SVN: Form matricies for numerical differentiation
    ################################################################
    def form_D_K(self):
        D_K = np.zeros((self.dim, self.dim))
        for d in range(self.DoF):
            D_K[d * self.nParticles : self.nParticles * (d + 1), d * self.nParticles : self.nParticles * (d + 1)] = self.kx
        return D_K / self.nParticles

    def form_HBD_for_numdiff(self, X):
        # Input is an m*b vector
        X = X.reshape(self.nParticles, self.DoF)
        kx, gkx = getGaussianKernelWithDerivatives_metric(X)
        GN_Hmlpt = self.stein.getGNHessianMinusLogPosterior_ensemble_new(X)
        h_ij_BD = self.stein.h_ij_BD(GN_Hmlpt, kx, gkx)
        H_BD = self.makeFullMatrixFromBlockDiagonal(h_ij_BD)
        return H_BD

    def form_H_for_numdiff(self, X):
        # Input is an m*b vector
        X = X.reshape(self.nParticles, self.DoF)
        kx, gkx = getGaussianKernelWithDerivatives_metric(X)
        GN_Hmlpt = self.stein.getGNHessianMinusLogPosterior_ensemble_new(X)
        H = self.stein.H_bar(GN_Hmlpt, kx, gkx)
        return H

    def form_Hposdef_for_numdiff(self, X):
        # Input is an m*b vector
        X = X.reshape(self.nParticles, self.DoF)
        kx, gkx = getGaussianKernelWithDerivatives_metric(X)
        GN_Hmlpt = self.stein.getGNHessianMinusLogPosterior_ensemble_new(X)
        H = self.stein.H_posdef(GN_Hmlpt, kx, gkx)
        return H

    def formK_for_numerical_derivative(self, X):
        # Input is an m*b vector
        X = X.reshape(self.nParticles, self.DoF)
        kx, gkx = getGaussianKernelWithDerivatives_metric(X)
        res = np.zeros((self.dim, self.dim))
        for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
            block = kx[m, n] * np.eye(self.DoF)
            res[m * self.DoF : self.DoF * (m + 1), n * self.DoF : self.DoF * (n + 1)] = block
        return res / self.nParticles

    def form_D_SVN_BD_for_numerical_derivative(self, X):
        # Input is an m*b vector
        X = X.reshape(self.nParticles, self.DoF)
        kx, gkx = getGaussianKernelWithDerivatives_metric(X)
        K = np.zeros((self.dim, self.dim))
        for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
            block = kx[m, n] * np.eye(self.DoF)
            K[m * self.DoF : self.DoF * (m + 1), n * self.DoF : self.DoF * (n + 1)] = block
        K = K / self.nParticles
        GN_Hmlpt = self.stein.getGNHessianMinusLogPosterior_ensemble_new(X)
        h_ij_BD = self.stein.h_ij_BD(GN_Hmlpt, kx, gkx)
        HBD_inv = tf.linalg.inv(h_ij_BD)
        HBD_inv_ndnd = self.makeFullMatrixFromBlockDiagonal(HBD_inv)
        return self.nParticles * K @ HBD_inv_ndnd @ K

    ###################################################
    # Auxilliary helper methods
    ###################################################
    def getBlock(self, mat, dim, m, n):
        # Works for both N x N blocks and D x D blocks.
        block = mat[m * dim : dim * (m + 1), n * dim : dim * (n + 1)]
        return block

    def makeFullMatrixFromBlockDiagonal(self, mbd):
        res = np.zeros((self.dim, self.dim))
        for m in range(self.nParticles):
            res[m * self.DoF : self.DoF * (m + 1), m * self.DoF : self.DoF * (m + 1)] = mbd[m]
        return res

    def makeFullMatrixFromBlockDiagonal_nxn_blocks(self, dnn):
        res = np.zeros((self.dim, self.dim))
        for d in range(self.DoF):
            res[d * self.nParticles : self.nParticles * (d + 1), d * self.nParticles : self.nParticles * (d + 1)] = dnn[d]
        return res

    def getP(self, N, D):
        dim = N * D
        P = np.zeros((dim, dim))
        for i in range(D):
            for m in range(N):
                P[i * N: (i + 1) * N, m * D : (m + 1) * D][m, i] = 1
        return P

    def test_getP(self):
        N = 12
        D = 11
        v = np.random.rand(N, D)
        vc = v.flatten()
        vh = v.flatten(order='F')
        P = self.getP(N=N, D=D)
        vh_test = P @ vc
        assert np.allclose(vh, vh_test)

    def test_getSVN_stc(self):
        b = np.random.normal(0, 1, self.dim)
        # Naive calculation
        D_K_sqrt = tf.linalg.sqrtm(self.D_K).numpy()
        H = self.makeFullMatrixFromBlockDiagonal(self.HBD)
        U = tf.linalg.cholesky(H).numpy().T
        P = self.getP(N=self.nParticles, D=self.DoF)
        H_inv_sample = tf.linalg.triangular_solve(U, b[..., np.newaxis], lower=False).numpy()
        v_stc_test = (np.sqrt(2) * P.T @ D_K_sqrt @ P @ H_inv_sample).reshape(self.nParticles, self.DoF)
        # Test
        kx_sqrt = tf.linalg.sqrtm(self.kx).numpy()
        LHBD = tf.linalg.cholesky(self.HBD).numpy()
        # UHBD = contract('mij -> mji', LHBD)
        v_stc = self.stein.getSVN_vstc(LHBD, kx_sqrt, B=b.reshape(self.nParticles, self.DoF))
        assert np.allclose(v_stc_test, v_stc)
        # D_k_sqrt_tmp = np.repeat(kx_sqrt[np.newaxis,...], self.DoF, axis=0) / np.sqrt(self.nParticles)
        # D_k_sqrt = self.makeFullMatrixFromBlockDiagonal_nxn_blocks(D_k_sqrt_tmp)

def main():
    a = Test_sanity()
    a.setup()
    a.test_getSVN_direction()
    # a.test_getSVN_Deterministic_correction()
    a.test_getSVN_BD_Deterministic_correction()
    a.test_getSVN_BD_Stochastic_correction()
    a.test_sSVN_uphill_numerical_divergence()
    a.test_getA_BD()
    a.test_getSVN_stc()

if __name__ is '__main__':
    main()