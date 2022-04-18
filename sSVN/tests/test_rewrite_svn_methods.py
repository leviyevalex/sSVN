import numpy as np
import itertools
from models.rosenbrock_analytic_proper_sample import rosenbrock_analytic
from source.stein_experiments import SVI
import os
import copy
import scipy

# t = timeit.Timer(functools.partial(self.stein.getGNHessianMinusLogPosterior_ensemble_new, self.stein.particles))
# t_contract = timeit.Timer(functools.partial(self.H_bar_contract, GN_Hmlpt, kernel, gradKernel))
class Test_sanity:
    def setup(self):
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/outdir'
        model = rosenbrock_analytic()
        # self.rtol = 1e-12
        # self.atol = 1e-12
        ###########################################################
        # Define and run for 1 iteration to instantiate all methods
        ###########################################################
        self.rtol = 1e-15
        self.atol = 1e-15
        self.optimizeMethod = 'SVN'
        self.nParticles = 4
        self.nIterations = 1
        self.stein = SVI(model = model, nParticles=self.nParticles, nIterations=self.nIterations, optimizeMethod = self.optimizeMethod)
        self.stein.SVGD_stochastic_correction = False
        self.nParticles_test = self.nParticles
        self.test_set_history = {}
        self.X = np.copy(self.stein.model.newDrawFromPrior(self.nParticles)) # copy because particles are updated after apply()
        self.T = self.stein.model.newDrawFromPrior(self.nParticles_test)
        self.DoF = self.stein.DoF
        self.stein.pickleData = False
        self.stein.constructMap()

        ##############################################################
        # Calculate all quantities to compare to naive implementations
        ##############################################################
        self.GNHmlpt_new = self.stein.getGNHessianMinusLogPosterior_ensemble_new(self.X)
        self.M = np.mean(self.GNHmlpt_new, axis=0)
        self.deltas = self.stein.getDeltas(self.X, self.X)
        self.deltas_T = self.stein.getDeltas(self.X, self.T)
        self.metricDeltas_T = self.stein.getMetricDeltas(self.M, self.deltas_T)
        self.metricDeltas = self.stein.getMetricDeltas(self.M, self.deltas)
        self.deltasMetricDeltas = self.stein.getDeltasMetricDeltas(self.deltas, self.metricDeltas)
        self.bandwidth = self.stein.getBandwidth(self.deltasMetricDeltas)
        self.kernel = self.stein.getKernelPiecewise(self.bandwidth, self.deltasMetricDeltas)
        self.gradKernel = self.stein.getGradKernelPiecewise(self.bandwidth, self.kernel, self.metricDeltas)
        self.Hesskx = self.stein.getHessianKernel(self.bandwidth, self.kernel, self.gradKernel, self.M, self.metricDeltas)
        self.xi = self.stein.getXi(self.bandwidth, self.metricDeltas, self.deltasMetricDeltas)['xi']
        self.gmlpt = self.stein.getGradientMinusLogPosterior_ensemble_new(self.X)
        self.GN_Hmlpt = self.stein.getGNHessianMinusLogPosterior_ensemble_new(self.X)
        self.mgJ = self.stein.mgJ_new(self.kernel, self.gradKernel, self.gmlpt)
        self.H_bar = self.stein.H_bar(self.GN_Hmlpt, self.kernel, self.gradKernel)
        self.alphas = self.stein.solveSystem_new(self.H_bar, self.mgJ)
        self.w = self.stein.w(self.alphas, self.kernel)
        self.grad_w = self.stein.grad_w_new(self.alphas, self.gradKernel)
        self.deltas_testMetricDeltas_test = self.stein.getDeltasMetricDeltas(self.deltas_T, self.metricDeltas_T)
        self.kernel_test = self.stein.getKernelPiecewise(self.bandwidth, self.deltas_testMetricDeltas_test)
        self.gradKernel_test = self.stein.getGradKernelPiecewise(self.bandwidth, self.kernel_test, self.metricDeltas_T)
        self.w_T = self.stein.w(self.alphas, self.kernel_test)
        self.grad_w_T = self.stein.grad_w_new(self.alphas, self.gradKernel_test)
        self.DK_nndd = self.stein.getSVGD_Diffusion(self.kernel)
        self.noise_step = 0.01
        # self.gradH = self.stein.gradH(self.GN_Hmlpt, self.kernel, self.gradKernel, self.Hesskx)
        self.gradH_BD = self.stein.grad_hij_BD_new(self.GN_Hmlpt, self.kernel, self.gradKernel, self.Hesskx)
        self.H_bar_BD = self.stein.H_bar_BD(self.GN_Hmlpt, self.kernel, self.gradKernel)
        # self.noise_correction = self.stein.getSVGD_noise_correction(self.kernel, self.noise_step)
    ############################################################
    # Stochastic SVN: Helper methods
    ############################################################
    def makeFullMatrixFromBlockDiagonal(self, mbd):
        dim = self.nParticles * self.DoF
        res = np.zeros((dim, dim))
        for m in range(self.nParticles):
            res[m * self.DoF : self.DoF * (m + 1), m * self.DoF : self.DoF * (m + 1)] = mbd[m]
        return res
    ############################################################
    def delta_test(self, ensemble1, ensemble2, m, n):
        """
        Computes the separation between particles m and n in ensemble1 and ensemble 2 respectively
        Args:
            ensemble1:
            ensemble2:
            m: particle belonging to ensemble 1
            n: particle belonging to ensemble 2

        Returns: separation (1-2)

        """
        return ensemble1[m] - ensemble2[n]

    def test_H_bar_BD(self):
        # test = self.stein.H_bar_BD(self.GN_Hmlpt, self.kernel, self.gradKernel)
        nnddH_bar = self.stein.reshapeNDNDtoNNDD(self.H_bar)
        test_b = (np.einsum('xij, xz -> zij', self.GN_Hmlpt, self.kernel ** 2) + np.einsum('xzi, xzj -> zij', self.gradKernel, self.gradKernel)) / self.nParticles
        for m in range(self.nParticles):
            assert(np.allclose(nnddH_bar[m, m], self.H_bar_BD[m], rtol=self.rtol, atol=self.atol))
            assert(np.allclose(nnddH_bar[m, m], test_b[m], rtol=self.rtol, atol=self.atol))
        pass



    def test_getDeltas(self):
        """
        Compares vectorized and naive implementation
        Notes: getDeltas should return n1 x n2 x d array.
        1) test if the shape of the array is as predicted.
        2) test if it returns what we expect it to return
        Returns:

        """
        assert(self.deltas_T.shape[0] == self.X.shape[0])
        assert(self.deltas_T.shape[1] == self.T.shape[0])
        for m, n in itertools.product(range(self.X.shape[0]), range(self.T.shape[0])):
            assert(np.allclose(self.deltas_T[m, n], self.delta_test(self.X, self.T, m, n), rtol=self.rtol, atol=self.atol))

    def metricDeltas_test(self, ensemble1, ensemble2, m, n):
        x = ensemble1[m]
        y = ensemble2[n]
        return self.M @ (x - y)

    def test_getMetricDeltas(self):
        for m, n in itertools.product(range(self.X.shape[0]), range(self.T.shape[0])):
            # print(self.metricDeltas_T[m, n])
            # print(self.metricDeltas_test(self.X, self.T, m, n))
            a = self.metricDeltas_T[m, n]
            b = self.metricDeltas_test(self.X, self.T, m, n)
            print(m, n)
            assert(np.allclose(a, b, rtol=self.rtol, atol=self.atol))
            # assert(np.allclose(self.metricDeltas_T[m, n], self.metricDeltas_test(self.X, self.T, m, n), rtol=self.rtol, atol=self.atol))
            pass

    def getKernel_test(self, ensemble1, ensemble2, m, n):
        """
        Particle x Particle image under metric
        Args:
            m: row
            n: col

        Returns: scalar

        """
        x = ensemble1[m]
        y = ensemble2[n]
        return np.exp(-(x - y).T @ self.M @ (x - y)/self.bandwidth)

    def test_SVGD_Diffusion(self):
        for m, n in itertools.product(range(self.X.shape[0]), range(self.T.shape[0])):
            assert(np.allclose(self.DK_nndd[m,n], self.kernel[m,n] / self.nParticles * np.eye(self.DoF), rtol=self.rtol, atol=self.atol))

    def test_SVGD_noise_correction(self):
        for m, n in itertools.product(range(self.X.shape[0]), range(self.T.shape[0])):
            np.sqrt(2 * self.noise_step * self.kernel[m, n])

    def test_getKernelPiecewise(self):
        for m, n in itertools.product(range(self.X.shape[0]), range(self.T.shape[0])):
            assert(np.allclose(self.kernel[m, n], self.getKernel_test(self.X, self.X, m, n), rtol=self.rtol, atol=self.atol))

    def test_getGradKernelPiecewise(self):
        for m, n in itertools.product(range(self.X.shape[0]), range(self.T.shape[0])):
            gradKernelTest = -2 / self.bandwidth * self.kernel[m, n] * self.metricDeltas[m, n]
            assert(np.allclose(self.gradKernel[m, n], gradKernelTest, rtol=self.rtol, atol=self.atol))

    def mgJ_individual_test(self, z):
        """
        Calculates mgJ for particle z
        Args:
            z: particle label (eg: 1, 2.)

        Returns: mgJ

        """
        a = 0
        b = 0
        for n in range(self.nParticles):
            a += -1 * self.kernel[n, z] * self.gmlpt[n]
            b += self.gradKernel[n, z]
            # sum += -1 * kernel[n, z] * gmlpt[n] + gradKernel[n, z]
        pass
        sum = a + b
        return (sum / self.nParticles).reshape(1, self.DoF)

    def test_mgJ(self):
        for m in range(self.X.shape[0]):
            assert(np.allclose(self.mgJ.reshape(self.nParticles, self.DoF)[m], self.mgJ_individual_test(m), rtol=self.rtol, atol=self.atol))

    def h_ij_test(self, m, n):
        """
        Calculate h_ij(m, n) with a naive loop
        Args:
            m: row in block matrix (integer)
            n: col in block matrix (integer)

        Returns: h_ij d x d matrix

        """
        oproduct_loop = np.zeros((self.DoF, self.DoF))
        gn_loop = np.zeros((self.DoF, self.DoF))
        for l in range(self.nParticles):  # For the outer products
            gn_loop += self.GN_Hmlpt[l] * self.kernel[l, m] * self.kernel[l, n]
            # oproduct_loop += np.outer(self.gradKernel[l, m], self.gradKernel[l, n]) # OLD HESSIAN
            oproduct_loop += np.outer(self.gradKernel[l, n], self.gradKernel[l, m]) # NEW HESSIAN
        pass
        return (gn_loop + oproduct_loop) / self.nParticles

    def test_H_bar(self):
        for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
            a = self.H_bar[m * self.DoF : self.DoF * (m + 1), n * self.DoF : self.DoF * (n + 1)]
            b = self.h_ij_test(m, n)
            assert(np.allclose(a, b, rtol=self.rtol, atol=self.atol))

    def w_individual_test(self, m):
        w_test = 0
        for n in range(self.nParticles):
            w_test += self.alphas[n] * self.kernel[n, m]
        return w_test # Note that this SHOULDN'T have a nParticles divisor!!!

    def test_w(self):
        for m in range(self.nParticles):
            assert(np.allclose(self.w[m], self.w_individual_test(m), rtol=self.rtol, atol=self.atol))

    def gradw_individual_test(self, m):
        gradw = 0
        for n in range(self.nParticles):
            gradw -= np.outer(self.alphas[n], self.gradKernel[n, m])
        return gradw

    def test_grad_w(self):
        for m in range(self.T.shape[0]):
            assert (np.allclose(self.grad_w[m], self.gradw_individual_test(m), rtol=self.rtol, atol=self.atol))

    def phi_naive_test(self, w, gradw, eps):
        I = np.eye(self.DoF)
        nParticles_test = self.T.shape[0]
        cost = 0
        repulse = 0
        for n in range(nParticles_test):
            cost += self.stein.getMinusLogPosterior_individual(self.T[n] + eps * w[n])
            if n == 0:
                sign0 = np.sign(np.linalg.det(I + eps * gradw[n]))
            det = np.linalg.det(I + eps * gradw[n])
            if np.sign(det) != sign0:
                return np.inf
            else:
                repulse -= np.log(det)
        phi = (cost + repulse) / nParticles_test
        return phi

    # def hessian_kernel_naive_test(self, h, kx, gkx, metric, metricDeltas):
    def hessian_kernel_naive_test(self, m, n):
        return (-2 / self.bandwidth) * (np.outer(self.gradKernel[m, n], self.metricDeltas[m, n, :]) + self.kernel[m, n] * self.M)

    def jacobian_map_SVGD_naive_test(self, m):
        sum = 0
        for n in range(self.nParticles):
            sum += np.outer(self.gradKernel[n, m], self.gmlpt[n]) - self.hessian_kernel_naive_test(n, m)
        return (1 / self.nParticles) * sum

    def stein_discrepancy_naive_test(self):
        sum = 0
        mgJ = self.mgJ
        if self.mgJ.shape[0] != self.nParticles:
            mgJ = mgJ.reshape(self.nParticles, self.DoF)
        for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
            sum += self.kernel[m, n] * np.inner(mgJ[m], mgJ[n])
        return sum

    def optimizer_cost_x2_test(self, hi, dict):
        h = 1 / hi
        return {'cost': (h - 3) ** 2}

    def laplacianKDE_term_naive_test(self):
        trace_hess = 0
        for d in range(self.DoF):
            trace_hess += self.Hesskx[:, :, d, d]
        sumn = 0
        for n in range(self.nParticles):
            sumn += trace_hess[:, n]
        return sumn

    def gradKx_Xi_term_naive_test(self):
        dotprod = 0
        for n in range(self.nParticles):
            for d in range(self.DoF):
                dotprod += self.gradKernel[:, n, d] * self.xi[n, d]
        return dotprod

    def form_gradK_first_slot(self):
        # Note that this does not work!
        # Define SVGD diffusion matrix K such that first slot of kernel is free variable, and the second is fixed.
        dim = self.nParticles * self.DoF
        res = np.zeros((dim, dim, dim))
        #############################################################
        # Get the gradient of augmented kernel matrix
        #############################################################
        for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
            zeros = np.zeros((self.nParticles, self.DoF))
            zeros[m] = copy.deepcopy(self.gradKernel[m, n])
            zeros = zeros.flatten()
            block = np.einsum('ij, z -> ijz', np.eye(self.DoF), zeros)
            res[m * self.DoF : self.DoF * (m + 1), n * self.DoF : self.DoF * (n + 1), :] = block
        res /= self.nParticles
        divK_a = np.mean(self.gradKernel, axis=0).flatten()
        divK_b = np.einsum('acc -> a', res)
        # The contribution to the trace from the first slot is zero!!!
        # try:
        assert np.allclose(divK_a, divK_b, rtol=self.rtol, atol=self.atol)

    def form_gradK_second_slot(self):
        # This one works!
        dim = self.nParticles * self.DoF
        res = np.zeros((dim, dim, dim))
        #############################################################
        # Get the gradient of augmented kernel matrix
        #############################################################
        for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
            zeros = np.zeros((self.nParticles, self.DoF))
            zeros[n] = copy.deepcopy(-1 * self.gradKernel[m, n]) # Negate because derivative on second slot.
            zeros = zeros.flatten()
            block = np.einsum('ij, z -> ijz', np.eye(self.DoF), zeros)
            res[m * self.DoF : self.DoF * (m + 1), n * self.DoF : self.DoF * (n + 1), :] = block
        res /= self.nParticles
        rep_a = np.mean(self.gradKernel, axis=0).flatten()
        rep_b = np.einsum('acc -> a', res)
        try:
            assert np.allclose(rep_a, rep_b, rtol=self.rtol, atol=self.atol)
            return res
        except:
            return None

    def formGradH_bar(self):
        dim = self.nParticles * self.DoF
        res = np.zeros((dim, dim, dim))
        #############################################################
        # Helper function to get the blocks for the matrix
        #############################################################
        def makeBlock(m, n):
            block = np.zeros((self.DoF, self.DoF, dim))
            for d, b in itertools.product(range(self.DoF), range(self.DoF)):
                entry = np.zeros((self.nParticles, self.DoF))
                entry[m] = copy.deepcopy(self.gradH[m, n, d, b, :])
                entry = entry.flatten()
                block[d, b, :] = entry
            return block
        #############################################################
        # Construct augmented grad H matrix explicitly
        #############################################################
        for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
            block = makeBlock(m, n)
            res[m * self.DoF : self.DoF * (m + 1), n * self.DoF : self.DoF * (n + 1), :] = block
        return res

    def formGradH_bar_BD(self):
        dim = self.nParticles * self.DoF
        res = np.zeros((dim, dim, dim))
        #############################################################
        # Helper function to get the blocks for the matrix
        #############################################################
        def makeBlock(m):
            block = np.zeros((self.DoF, self.DoF, dim))
            for d, b in itertools.product(range(self.DoF), range(self.DoF)):
                entry = np.zeros((self.nParticles, self.DoF))
                entry[m] = copy.deepcopy(self.gradH_BD[m, d, b, :])
                entry = entry.flatten()
                block[d, b, :] = entry
            return block
        #############################################################
        # Construct augmented grad H matrix explicitly
        #############################################################
        for m in range(self.nParticles):
            block = makeBlock(m)
            res[m * self.DoF : self.DoF * (m + 1), m * self.DoF : self.DoF * (m + 1), :] = block
        return res

    def formK(self):
        # X = X.reshape(self.nParticles, self.DoF)
        # deltas = self.stein.getDeltas(X, X)
        # metricDeltas = self.stein.getMetricDeltas(metric, deltas)
        # deltaMetricDeltas = self.stein.getDeltasMetricDeltas(deltas, metricDeltas)
        # kx = self.stein.getKernelPiecewise(h, deltaMetricDeltas)
        dim = self.nParticles * self.DoF
        res = np.zeros((dim, dim))
        for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
            block = self.kernel[m, n] * np.eye(self.DoF)
            res[m * self.DoF : self.DoF * (m + 1), n * self.DoF : self.DoF * (n + 1)] = block
        return res / self.nParticles






    def form_gradK_manually(self, X, h, metric):
        # X = X.reshape(self.nParticles, self.DoF)
        dim = self.nParticles * self.DoF
        res = np.zeros((dim, dim, dim))
        deltas = self.stein.getDeltas(X, X)
        metricDeltas = self.stein.getMetricDeltas(metric, deltas)
        deltaMetricDeltas = self.stein.getDeltasMetricDeltas(deltas, metricDeltas)
        kx = self.stein.getKernelPiecewise(h, deltaMetricDeltas)
        gkx = self.stein.getGradKernelPiecewise(h, kx, metricDeltas)
        #############################################################
        # Get the gradient of augmented kernel matrix
        #############################################################
        for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
            zeros = np.zeros((self.nParticles, self.DoF))
            zeros[n] = copy.deepcopy(gkx[m, n])
            zeros = zeros.flatten()
            block = np.einsum('ij, z -> ijz', np.eye(self.DoF), zeros)
            res[m * self.DoF : self.DoF * (m + 1), n * self.DoF : self.DoF * (n + 1), :] = block
        res /= -1 * self.nParticles
        divK_a = np.mean(gkx, axis=0).flatten()
        divK_b = np.einsum('acc -> a', res)
        try:
            assert np.allclose(divK_a, divK_b, rtol=self.rtol, atol=self.atol)
            return res
        except:
            return None





    def gradK_action_naive_BD(self):
        dim = self.nParticles * self.DoF
        res = self.form_gradK_second_slot()
        if res is None:
            raise Exception('ERROR: gradK formed does not match SVGD')
        #################################################################################
        # Get relevent matricies
        #################################################################################
        Bnndd = np.random.rand(self.nParticles, self.nParticles, self.DoF, self.DoF)
        Bndd = np.einsum('nndb -> ndb', Bnndd)
        Bndnd = Bnndd.swapaxes(1, 2).reshape(dim, dim)
        vec_naive = np.einsum('abc, bc -> a', res, Bndnd)
        #################################################################################
        # Smart way to get all the blocks
        #################################################################################
        # test_one = self.stein.gradK_action(self.gradKernel, Bnndd).flatten()
        test_two = self.stein.gradK_action_BD(self.gradKernel, Bndd).flatten()
        # assert np.allclose(test_one, vec_naive, rtol=self.rtol, atol=self.atol)
        assert np.allclose(test_two, vec_naive, rtol=self.rtol, atol=self.atol)


    # def test_gradK_action_BD_mat(self):
    #     dim = self.nParticles * self.DoF
    #     gradK = self.form_gradK()
    #     if gradK is None:
    #         raise Exception('ERROR: gradK formed does not match SVGD')
    #     #################################################################################
    #     # Get relevent matricies
    #     #################################################################################
    #     Bndd = np.random.rand(self.nParticles, self.DoF, self.DoF)
    #     Bndnd = np.zeros((dim, dim))
    #     for m in range(self.nParticles):
    #         Bndnd[m * self.DoF : self.DoF * (m + 1), m * self.DoF : self.DoF * (m + 1)] = Bndd[m]
    #     vec_naive = np.einsum('abc, bc -> a', gradK, Bndnd)
    #     #################################################################################
    #     # Naive way to get first a = 0
    #     #################################################################################
    #     mat_a = np.zeros((self.nParticles, self.DoF, self.DoF))
    #     k = self.gradKernel[:, 0, :]
    #     for n in range(self.nParticles):
    #         temp = np.zeros((self.DoF, self.DoF))
    #         temp[0] = k[n]
    #         mat_a[n, :, :] =  temp @ Bbd[n].T
    #     test_output = np.mean(mat_a[:, 0, 0])
    #     assert np.allclose(test_output, vec_naive[0], rtol=self.rtol, atol=self.atol)
    #     #################################################################################
    #     # Smart way to get first a block
    #     #################################################################################
    #     np.mean(np.einsum('nbd, nd -> nb', Bbd, k), axis=0)
    #     #################################################################################
    #     # Smart way to get all the blocks
    #     #################################################################################
    #     vec_smart_a = np.mean(np.einsum('nbd, nmd -> nmb', Bbd, self.gradKernel), axis=0).flatten()
    #     ################################################################################
    #     # Smart way to get all the blocks
    #     ################################################################################
    #     # test = self.stein.gradK_action(self.gradKernel, Bnndd).flatten()
    #     # assert np.allclose(test, vec_naive, rtol=self.rtol, atol=self.atol)



    # def test_gradH_action_BLOCK_DIAG(self):
    #     np.random.seed(1)
    #     dim = self.nParticles * self.DoF
    #     Bndd = np.random.rand(self.nParticles, self.DoF, self.DoF)
    #     Bndnd = np.zeros((dim, dim))
    #     for m in range(self.nParticles):
    #         Bndnd[m * self.DoF : self.DoF * (m + 1), m * self.DoF : self.DoF * (m + 1)] = Bndd[m]
    #     res = self.formGradH_bar_BD()
    #     vec_naive = np.einsum('abc, bc -> a', res, Bndnd)
    #     # test_d = np.einsum('mnabc, nmbc -> ma', gradH, Bndd).flatten()
    #     # test = np.einsum('mmabc, mbc -> ma', self.gradH, Bndd).flatten()
    #     test = self.stein.gradH_action_BD(self.gradH_BD, Bndd).flatten()
    #     # test = self.stein.gradH_action(self.gradH, Bnndd).flatten()
    #     assert np.allclose(test, vec_naive, rtol=1e-14, atol=1e-14)

    def test_phi(self):
        phi0 = self.stein.phi_new(self.T, self.w_T, self.grad_w_T, 0)
        phi1 = self.stein.phi_new(self.T, self.w_T, self.grad_w_T, 1)
        assert np.allclose(phi0, self.phi_naive_test(self.w_T, self.grad_w_T, 0), rtol=self.rtol, atol=self.atol)
        assert np.allclose(phi1, self.phi_naive_test(self.w_T, self.grad_w_T, 1), rtol=self.rtol, atol=self.atol)

    def test_hessian_kernel(self):
        for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
            a = self.hessian_kernel_naive_test(m, n)
            b = self.Hesskx[m, n]
            # b = self.stein.getHessianKernel(self.bandwidth, self.kernel, self.gradKernel, self.M, self.metricDeltas)[m, n]
            assert(np.allclose(a, b, rtol=self.rtol, atol=self.atol))

    def test_jacobian_map_SVGD(self):
        for m in range(self.nParticles):
            a = self.jacobian_map_SVGD_naive_test(m)
            b = self.stein.getJacobianMapSVGD(self.bandwidth, self.kernel, self.gradKernel, self.M, self.metricDeltas, self.gmlpt, hesskx=None)[m]
            assert(np.allclose(a, b, rtol=self.rtol, atol=self.atol))

    def test_stein_discrepency(self):
        a = self.stein.getSteinDiscrepancy(self.kernel, self.mgJ)
        b = self.stein_discrepancy_naive_test()
        assert(np.allclose(a, b, rtol=self.rtol, atol=self.atol))

    def test_three_point_optimize(self):
        cost = self.optimizer_cost_x2_test
        input_dict = {}
        for i in range(300):
            if i == 0:
                a_prev = 5
            else:
                a_prev = a
            a = self.stein.three_point_quadratric_optimize_NEW(a_prev, cost, {})['h']
        assert(np.allclose(a, 3, rtol=1e-4, atol=1e-4))

    def test_laplacianKDE_term(self):
        a = self.stein.HE_laplacianKDE_term(self.Hesskx)
        b = self.laplacianKDE_term_naive_test()
        assert(np.allclose(a, b, rtol=self.rtol, atol=self.atol))

    def test_gradKx_Xi_term(self):
        a = self.stein.HE_gradKx_Xi_term(self.gradKernel, self.xi)
        b = self.gradKx_Xi_term_naive_test()
        assert(np.allclose(a, b, rtol=self.rtol, atol=self.atol))
    ######################################################
    # Stochastic SVN contraction tests
    ######################################################
    # (I)



    # def test_gradH_action_mnbd(self):
    #     np.random.seed(1)
    #     dim = self.nParticles * self.DoF
    #     Bnndd = np.random.rand(self.nParticles, self.nParticles, self.DoF, self.DoF)
    #     Bndnd = Bnndd.swapaxes(1, 2).reshape(dim, dim)
    #     res = self.formGradH_bar()
    #     vec_naive = np.einsum('abc, bc -> a', res, Bndnd)
    #     test = self.stein.gradH_action_mnbd(self.gradH, Bnndd).flatten()
    #     # test = self.stein.gradH_action_mnbd(self.gradH, Bnndd).flatten()
    #     assert np.allclose(test, vec_naive, rtol=1e-14, atol=1e-14)

    # (V)


    ######################################################
    # End stochastic svn methods
    ######################################################





    # def gradK_naive_test_scratchwork(self):
    #     dim = self.nParticles * self.DoF
    #     res = np.zeros((dim, dim, dim))
    #     #############################################################
    #     # Get the gradient of augmented kernel matrix
    #     #############################################################
    #     for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
    #         zeros = np.zeros((self.nParticles, self.DoF))
    #         zeros[n] = copy.deepcopy(self.gradKernel[m, n])
    #         zeros = zeros.flatten()
    #         block = np.einsum('ij, z -> ijz', np.eye(self.DoF), zeros)
    #         res[m * self.DoF : self.DoF * (m + 1), n * self.DoF : self.DoF * (n + 1), :] = block
    #     res /= -1 * self.nParticles
    #     # i) Minus sign accounts for divergence on second slot
    #     #################################################################################
    #     # Check that the divergence of augmented kernel matrix agrees with SVGD repulsion
    #     #################################################################################
    #     divK_a = np.mean(self.gradKernel, axis=0).flatten()
    #     divK_b = np.einsum('acc -> a', res)
    #     assert np.allclose(divK_a, divK_b, rtol=self.rtol, atol=self.atol)
    #     #################################################################################
    #     # Get relevent matricies
    #     #################################################################################
    #     # mat = np.zeros((self.DoF, self.DoF))
    #     Bnndd = np.random.rand(self.nParticles, self.nParticles, self.DoF, self.DoF)
    #     Bndnd = Bnndd.swapaxes(1, 2).reshape(dim, dim)
    #     Bbd = np.einsum('nnbd -> nbd', Bnndd)
    #     vec_naive = np.einsum('abc, bc -> a', res, Bndnd)
    #     #################################################################################
    #     # Naive way to get first a = 0
    #     #################################################################################
    #     mat_a = np.zeros((self.nParticles, self.DoF, self.DoF))
    #     k = self.gradKernel[:, 0, :]
    #     for n in range(self.nParticles):
    #         temp = np.zeros((self.DoF, self.DoF))
    #         temp[0] = k[n]
    #         mat_a[n, :, :] =  temp @ Bbd[n].T
    #     test_output = np.mean(mat_a[:, 0, 0])
    #     assert np.allclose(test_output, vec_naive[0], rtol=self.rtol, atol=self.atol)
    #     #################################################################################
    #     # Smart way to get first a block
    #     #################################################################################
    #     np.mean(np.einsum('nbd, nd -> nb', Bbd, k), axis=0)
    #     #################################################################################
    #     # Smart way to get all the blocks
    #     #################################################################################
    #     vec_smart_a = np.mean(np.einsum('nbd, nmd -> nmb', Bbd, self.gradKernel), axis=0).flatten()
    #     vec_smart_b = np.einsum('nbd, nmd -> mb', Bbd, self.gradKernel).flatten() / self.nParticles
    #     vec_smart_c = np.einsum('nnbd, nmd -> mb', Bnndd, self.gradKernel).flatten() / self.nParticles
    #     assert np.allclose(vec_smart_a, vec_naive, rtol=self.rtol, atol=self.atol)
    #     assert np.allclose(vec_smart_b, vec_naive, rtol=self.rtol, atol=self.atol)
    #     assert np.allclose(vec_smart_c, vec_naive, rtol=self.rtol, atol=self.atol)
    #     pass

    # def gradH_bar_naive_test(self):
    #     np.random.seed(1)
    #     dim = self.nParticles * self.DoF
    #     res = np.zeros((dim, dim, dim))
    #     Bnndd = np.random.rand(self.nParticles, self.nParticles, self.DoF, self.DoF)
    #     Bndnd = Bnndd.swapaxes(1, 2).reshape(dim, dim)
    #     #############################################################
    #     # Helper function to get the blocks for the matrix
    #     #############################################################
    #     def makeBlock(m, n):
    #         block = np.zeros((self.DoF, self.DoF, dim))
    #         for d, b in itertools.product(range(self.DoF), range(self.DoF)):
    #             entry = np.zeros((self.nParticles, self.DoF))
    #             entry[m] = copy.deepcopy(self.gradH[m, n, d, b, :])
    #             entry = entry.flatten()
    #             block[d, b, :] = entry
    #         return block
    #     #############################################################
    #     # Construct augmented grad H matrix explicitly
    #     #############################################################
    #     for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
    #         block = makeBlock(m, n)
    #         res[m * self.DoF : self.DoF * (m + 1), n * self.DoF : self.DoF * (n + 1), :] = block
    #     #############################################################
    #     # Get action naively
    #     #############################################################
    #     vec_naive = np.einsum('abc, bc -> a', res, Bndnd)
    #     #############################################################
    #     # a = 0 check
    #     #############################################################
    #     test_a = np.trace(res[0] @ Bndnd.T)
    #     assert np.allclose(test_a, vec_naive[0], rtol=self.rtol, atol=self.atol)
    #     temp = self.gradH[0, :, 0, :, :]
    #     sum = 0
    #     for n in range(self.nParticles):
    #         sum += np.trace(temp[n] @ Bnndd[n, 0].T)
    #     assert np.allclose(sum, vec_naive[0], rtol=self.rtol, atol=self.atol)
    #     #############################################################
    #     # Get action in a smart way for a = 0
    #     #############################################################
    #     Bcols = Bnndd[:,0]
    #     test_b = np.einsum('ndb, ndb -> ', temp, Bcols)
    #     assert np.allclose(test_b, vec_naive[0], rtol=self.rtol, atol=self.atol)
    #     #############################################################
    #     # Get action in a smart way for first block
    #     #############################################################
    #     np.einsum('nabc, nbc -> a', self.gradH[0], Bnndd[:,0])
    #     #############################################################
    #     # Get action in a smart way for all block
    #     #############################################################
    #     test_d = np.einsum('mnabc, nmbc -> ma', self.gradH, Bnndd).flatten()
    #     # agrees up to 14 digits!
    #     assert np.allclose(test_d, vec_naive, rtol=1e-14, atol=1e-14)
    #     pass

    def test_Hess_likelihood(self):
        import numdifftools as nd
        a = np.array([0.1, 0.2])
        hess_numerical = nd.Hessian(self.stein.model.getMinusLogLikelihood_individual)(a)
        hess_analytic = self.stein.model.getHessianMinusLogLikelihood_individual(a)
        assert(np.allclose(hess_numerical, hess_analytic, rtol=1e-5, atol=1e-5))

    def test_Hess_posterior(self):
        import numdifftools as nd
        a = np.array([0.1, 0.2])
        hess_numerical = nd.Hessian(self.stein.getMinusLogPosterior_individual)(a)
        hess_analytic = self.stein.getHessianMinusLogPosterior_individual(a)
        assert(np.allclose(hess_numerical, hess_analytic, rtol=1e-5, atol=1e-5))




    #############################################################################
    #############################################################################
    #############################################################################

    ##########################################################
    # Stochastic SVN Block Diagonal: Contraction method tests
    ##########################################################
    # 1
    def test_K_action_mb_vec(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.zeros((2, 2))
        b[0] = np.array([1, 2])
        b[1] = np.array([1, 3])
        ans = np.array([[3, 8], [7, 18]])
        test = self.stein.K_action_mb_vec(a, b) * self.nParticles
        assert(np.allclose(ans, test, rtol=self.rtol, atol=self.atol))
    # 2
    def test_K_action_mbd_mat(self):
        dim = self.nParticles * self.DoF
        K = self.formK()
        Bndnd = np.zeros((dim, dim))
        Bnd = np.random.rand(self.nParticles, self.DoF, self.DoF)
        for m in range(self.nParticles):
            Bndnd[m * self.DoF : self.DoF * (m + 1), m * self.DoF : self.DoF * (m + 1)] = Bnd[m]
        test_a = K @ Bndnd
        # test_b = self.stein.reshapeNNDDtoNDND(np.einsum('nm, nbd -> mnbd', self.kernel, Bnd))/self.nParticles
        test_b = self.stein.reshapeNNDDtoNDND(self.stein.K_action_mbd_mat(self.kernel, Bnd))
        assert(np.allclose(test_a, test_b, rtol=self.rtol, atol=self.atol))
    # 3
    def test_K_action_mnbd_mat(self):
        dim = self.nParticles * self.DoF
        K = self.formK()
        Bndnd = np.random.rand(dim, dim)
        Bnndd = self.stein.reshapeNDNDtoNNDD(Bndnd)
        test_a = K @ Bndnd
        # test_b = self.stein.reshapeNNDDtoNDND(np.einsum('mo, onbd -> mnbd', self.kernel, Bnndd)) / self.nParticles
        test_b = self.stein.reshapeNNDDtoNDND(self.stein.K_action_mnbd_mat(self.kernel, Bnndd))
        assert(np.allclose(test_a, test_b, rtol=self.rtol, atol=self.atol))
        pass
    # 4 # OUTDATED MAKE SURE TO REMOVE LATER
    def test_gradK_action_mnbd(self):
        dim = self.nParticles * self.DoF
        res = self.form_gradK_second_slot()
        if res is None:
            raise Exception('ERROR: gradK formed does not match SVGD')
        #################################################################################
        # Get relevent matricies
        #################################################################################
        Bnndd = np.random.rand(self.nParticles, self.nParticles, self.DoF, self.DoF)
        # Bndd = np.einsum('nndb -> ndb', Bnndd)
        Bndnd = Bnndd.swapaxes(1, 2).reshape(dim, dim)
        vec_naive = np.einsum('abc, bc -> a', res, Bndnd)
        #################################################################################
        # Smart way to get all the blocks
        #################################################################################
        test_one = self.stein.gradK_action_mnbd(self.gradKernel, Bnndd).flatten()
        # test_two = self.stein.gradK_action_BD(self.gradKernel, Bndd).flatten()
        assert np.allclose(test_one, vec_naive, rtol=self.rtol, atol=self.atol)
        # assert np.allclose(test_two, vec_naive, rtol=self.rtol, atol=self.atol)
    # 5
    def test_gradHBD_mnbd(self):
        np.random.seed(1)
        dim = self.nParticles * self.DoF
        Bnndd = np.random.rand(self.nParticles, self.nParticles, self.DoF, self.DoF)
        Bndnd = self.stein.reshapeNNDDtoNDND(Bnndd)
        res = self.formGradH_bar_BD()
        test_a = np.einsum('abc, bc -> a', res, Bndnd)
        # First entry in vector
        a = np.trace(self.gradH_BD[0, 0, :, :] @ Bnndd[0, 0].T)
        # Rewritten in einsum form
        b = np.einsum('db, db -> ', self.gradH_BD[0, 0, :, :], Bnndd[0, 0])
        # First block
        c = np.einsum('adb, db -> a', self.gradH_BD[0, :, :, :], Bnndd[0, 0])
        # All blocks
        d = np.einsum('nadb, nndb -> na', self.gradH_BD, Bnndd).flatten()
        test_b = self.stein.gradHBD_mnbd(self.gradH_BD, Bnndd).flatten()
        assert np.allclose(test_a, test_b, rtol=1e-14, atol=1e-14)
        pass
    # 6
    def test_mnbd_mb_matvec(self):
        A = np.random.rand(self.DoF, self.DoF)
        B = 2 * np.random.rand(self.DoF, self.DoF)
        BD = scipy.linalg.block_diag(A, B)
        vec = np.random.rand(2, self.DoF)
        test_a = self.stein.mnbd_mb_matvec(self.stein.reshapeNDNDtoNNDD(BD), vec).flatten()
        test_b = BD @ vec.flatten()
        assert np.allclose(test_a, test_b, rtol=self.rtol, atol=self.atol)
    # 7
    def test_mbd_mb_matvec(self):
        A = np.random.rand(self.DoF, self.DoF)
        B = 2 * np.random.rand(self.DoF, self.DoF)
        BD = scipy.linalg.block_diag(A, B)
        Bndd = np.zeros((2, self.DoF, self.DoF))
        Bndd[0] = A
        Bndd[1] = B
        vec = np.random.rand(2, self.DoF)
        test_a = self.stein.mbd_mb_matvec(Bndd, vec).flatten()
        test_b = BD @ vec.flatten()
        assert np.allclose(test_a, test_b, rtol=self.rtol, atol=self.atol)

    ##########################################################
    # Stochastic SVN Block Diagonal: Calculation method tests
    ##########################################################
    def test_gradHBD(self):
        for z, i, j, k in itertools.product(range(self.nParticles), range(self.DoF), range(self.DoF), range(self.DoF)):
            res = np.mean(-2 * self.GNHmlpt_new[:, i, j] * self.kernel[:,z] * self.gradKernel[:,z,k] - self.Hesskx[:, z, i, k] * self.gradKernel[:,z,j] - self.gradKernel[:,z,i] * self.Hesskx[:,z,j,k])
            assert np.allclose(res, self.gradH_BD[z,i,j,k], rtol=self.rtol, atol=self.atol)


    ##########################################################
    # Stochastic SVN Block Diagonal: Numerical tests
    ##########################################################
    def formK_for_numerical_derivative(self, X):
        # Input is an m*b vector
        X = X.reshape(self.nParticles, self.DoF)
        deltas = self.stein.getDeltas(X, X)
        metricDeltas = self.stein.getMetricDeltas(self.M, deltas)
        deltaMetricDeltas = self.stein.getDeltasMetricDeltas(deltas, metricDeltas)
        kx = self.stein.getKernelPiecewise(self.bandwidth, deltaMetricDeltas)
        dim = self.nParticles * self.DoF
        res = np.zeros((dim, dim))
        for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
            block = kx[m, n] * np.eye(self.DoF)
            res[m * self.DoF : self.DoF * (m + 1), n * self.DoF : self.DoF * (n + 1)] = block
        return res / self.nParticles

    def gradK_single_parameter_def(self):
        dim = self.nParticles * self.DoF
        res = np.zeros((dim, dim, dim))
        #############################################################
        # Get the gradient of augmented kernel matrix
        #############################################################
        for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
            zeros = np.zeros((self.nParticles, self.DoF))
            zeros[m] = copy.deepcopy(self.gradKernel[m, n])
            zeros[n] = copy.deepcopy(-1 * self.gradKernel[m, n])
            zeros = zeros.flatten()
            block = np.einsum('ij, z -> ijz', np.eye(self.DoF), zeros)
            res[m * self.DoF : self.DoF * (m + 1), n * self.DoF : self.DoF * (n + 1), :] = block
        res /= self.nParticles
        rep_a = np.mean(self.gradKernel, axis=0).flatten()
        rep_b = np.einsum('acc -> a', res)
        try:
            assert np.allclose(rep_a, rep_b, rtol=self.rtol, atol=self.atol)
            return res
        except:
            return None

    #################################################################
    # Stochastic SVN methods: Testing new methods
    #################################################################
    def test_agreement_gradK_numerical_analytic(self):
        # test_a = self.formK_for_numerical_derivative(X_flattened)
        # grad_regular = self.form_gradK_manually(self.X, self.bandwidth, self.M)
        X_flattened = self.X.flatten()
        import numdifftools as nd
        grad_manual = self.gradK_single_parameter_def()
        grad_numerical = nd.Gradient(self.formK_for_numerical_derivative)(X_flattened)
        rep_a = np.mean(self.gradKernel, axis=0).flatten()
        rep_b = np.einsum('acc -> a', grad_numerical)
        rep_c = np.einsum('acc -> a', grad_manual)
        grad_numerical = np.einsum('ijk -> kij', grad_numerical)
        rep_d = np.einsum('acc -> a', grad_numerical)
        assert np.allclose(grad_manual, grad_numerical, rtol=1e-6, atol=1e-6)
        pass
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
        gradK = self.gradK_single_parameter_def()
        testing_a = f_to_c(np.einsum('abc, bc -> a', gradD_K, mat_c_to_f(A)))
        testing_b = np.einsum('abc, bc -> a', gradK, A)
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
    # def test_gradK_action_new(self):
    #     A_DDNN = self.stein.reshapeNDNDtoDDNN(mat_c_to_f(A))
    #     gradD_K_block = (gkx_block / self.nParticles).reshape(self.nParticles, self.nParticles, 1, self.DoF, self.nParticles).swapaxes(1,3).reshape(self.nParticles, 1, self.DoF, self.nParticles, self.nParticles)
    #     tmp_c = np.einsum('obdmn, qdmn -> oq', gradD_K_block, A_DDNN).flatten()
    #     assert np.allclose(tmp_c, testing_b, rtol=1e-6, atol=1e-6)
    #     pass




# For debugging purposes
def main():
    a = Test_sanity()
    a.setup()
    # a.test_getMetricDeltas()
    # a.test_SVGD_Diffusion()
    # a.gradK_naive_test()
    # a.gradH_bar_naive_test()
    # a.test_Hess_dbl_rosen()
    # a.test_agreement_gradK_numerical_analytic()
    # a.test_gradH_action_BLOCK_DIAG()
    # a.test_BD_matvec()
    # a.test_K_action_mnbd_mat()
    # a.test_mnbd_mb_matvec()
    # a.test_gradH_action_mnbd()
    # a.form_gradK_first_slot()
    # a.test_gradHBD_mnbd()
    # a.gradK_single_parameter_def()
    # a.form_gradK_first_slot()
    # a. test_agreement_gradK_numerical_analytic()
    a.test_permutation_orders()
    # a.test_H_bar_BD()
    # a.test_gradH_BD()
    # a.test_gradK_action_BD_mat()
    # a.test_getDeltas()
    # a.test_metric()
    # a.test_getGradKernelPiecewise()
    # a.test_mgJ()
    # a.test_getMetricDeltas()
    # a.test_hessian_kernel()
    # a.test_jacobian_map_SVGD()
    # a.test_H_bar()
    # a.test_stein_discrepency()
    # a.test_three_point_optimize()
    # a.test_laplacianKDE_term()
    # a.test_gradKx_Xi_term()
    # a.test_w()
    # a.test_grad_w()
    # a.test_phi()
    # a.test_getKernelWhole()
if __name__ is '__main__':
    main()