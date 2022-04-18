import numpy as np
import pytest
import itertools
from models.gauss_analytic import gauss_analytic
from source.stein_experiments import SVI
import os


class Test_sanity:
    def setup(self):
        # Chose to redeclare. Can take any method and stick back into main methods easily if need be.
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/outdir'
        model = gauss_analytic()
        self.rtol = 1e-15
        self.atol = 1e-15
        self.optimizeMethod = 'SVN'
        self.nParticles = 3
        self.nIterations = 1
        self.stein = SVI(model = model, nParticles=self.nParticles, nIterations=self.nIterations, optimizeMethod = self.optimizeMethod, output_dir=output_dir)
        self.particles = np.copy(self.stein.particles) # copy because particles are updated after apply()
        self.DoF = self.stein.DoF
        self.stein.apply()
        self.gradient_k_gram = self.stein.gradient_k_gram
        self.GN_Hmlpt = self.stein.GN_Hmlpt
        self.gmlpt = self.stein.gmlpt
        self.M = self.stein.M
        self.H = self.stein.H
        self.k_gram = self.stein.k_gram
        self.bandwidth = self.stein.bandwidth
        self.alphas = self.stein.alphas


    def metric_action_individual(self, r, c):
        """
        Particle x Particle image under metric
        Args:
            r: row
            c: col

        Returns: scalar

        """
        x = self.particles[:, r]
        y = self.particles[:, c]
        return (x - y).T @ self.M @ (x - y)

    def metric_action_ensemble_test(self):
        """
        Computes image of metric over ensemble in plain math

        """
        metric_action = np.zeros((self.nParticles, self.nParticles))
        for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
            metric_action[m, n] = self.metric_action_individual(m, n)
        return metric_action

    def grad_kern_individual(self, r, c):
        """
        Computes analytic derivative of kernel
        Args:
            r: row
            c: col

        Returns:

        """
        x = self.particles[:, r]
        y = self.particles[:, c]
        return self.k_gram[r, c] * (-2 * self.M @ (x - y) / self.bandwidth)

    def grad_kern_ensemble_test(self):
        """
        Assembles object as used in algorithm for direct comparison

        """
        grad_kern = np.zeros((self.nParticles, self.nParticles, self.DoF))
        for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
            grad_kern[m , n, :] = self.grad_kern_individual(m, n)
        return grad_kern

    def get_mgJ_individual(self, z):
        """
        Calculates mgJ for particle z
        Args:
            z: particle label (eg: 1, 2.)

        Returns: mgJ

        """
        sum = 0
        for n in range(self.nParticles):
            sum += -1 * self.k_gram[n, z] * self.gmlpt[:, n] + self.grad_kern_individual(n, z)
        return (sum / self.nParticles).reshape(self.DoF, 1)

    def get_mgJ_ensemble_test(self):
        """
        Assembles mgJ for ensemble position update, or H solve.
        Returns: mgJ shaped appropriately

        """
        a = self.get_mgJ_individual(0)
        if self.optimizeMethod == 'SVGD':
            for n in range(1, self.nParticles):
                a = np.hstack((a, self.get_mgJ_individual(n)))
        elif self.optimizeMethod == 'SVN':
            for n in range(1, self.nParticles):
                a = np.vstack((a, self.get_mgJ_individual(n)))
        return a

    def calculateBlockLoop_test(self, m, n):
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
            gn_loop += self.GN_Hmlpt[:, :, l] * self.k_gram[l, m] * self.k_gram[l, n]
            oproduct_loop += np.outer(self.gradient_k_gram[l, m, :], self.gradient_k_gram[l, n, :])
        pass
        return (gn_loop + oproduct_loop) / self.nParticles

    def contract_alphaKernel_test(self):
        """
        Checks synthesis in buildEnsembleUpdate
        Returns: SVN step

        """
        w = np.zeros((self.DoF, self.nParticles))
        for z in range(self.nParticles):
            w_ind = np.zeros(self.DoF)
            for n in range(self.nParticles):
                w_ind = w_ind + self.alphas[:, n] * self.k_gram[n, z]
            w[:, z] = w_ind
        return w

    def grad_w_test(self, m):
        """
        Calculates gradient of search direction in plain math
        This measures how the flow direction for particle m changes as we independently vary the position of particle m.
        Args:
            m: particle number

        Returns: d x d matrix

        """
        sum = 0
        z = self.particles[:, m]
        for n in range(self.nParticles):
            xn = self.particles[:, n]
            sum += np.outer(self.alphas[:, n], (2 / self.bandwidth) * self.k_gram[n, m] * self.M @ (xn - z))
        return sum

    def test_metric(self):
        """
        Check if naive and vectorized metric action over ensemble match

        """
        assert(np.allclose(self.stein.metric_ensemble, self.metric_action_ensemble_test(), rtol=self.rtol, atol=self.atol))

    def test_grad_kernel(self):
        """
        Check if naive and vectorized \nabla k(x,y) match

        """
        assert(np.allclose(self.stein.gradient_k_gram, self.grad_kern_ensemble_test(), rtol=self.rtol, atol=self.atol))

    def test_build_mgJ(self):
        """
        Check if naive and vectorized mgJ match

        """
        assert(np.allclose(self.stein.mgJ_ensemble, self.get_mgJ_ensemble_test(), rtol=self.rtol, atol=self.atol))

    def test_H(self):
        """
        Check if:
        a) naive and vectorized h_ij blocks match
        b) full hessian (built from the blocks) are ordered properly

        """
        for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
            block = self.H[m * self.DoF : self.DoF * (m + 1), n * self.DoF : self.DoF * (n + 1)]
            assert(np.allclose(block, self.calculateBlockLoop_test(m, n), rtol=self.rtol, atol=self.atol))

    def test_SVN_step_synthesis(self):
        """
        Check if naive synthesis matches with vectorized implementation

        """
        assert(np.allclose(self.stein.Q, self.contract_alphaKernel_test(), rtol=self.rtol, atol=self.atol))

    def test_grad_w(self):
        for m in range(self.nParticles):
            assert(np.allclose(self.stein.grad_w(m), self.grad_w_test(m), rtol=self.rtol, atol=self.atol))

# For debugging purposes
def main():
    a = Test_sanity()
    a.setup()
    a.test_grad_w()
if __name__ is '__main__':
    main()

