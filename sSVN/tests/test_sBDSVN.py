import numpy as np
import itertools
from models.rosenbrock_analytic_proper_sample import rosenbrock_analytic
from source.stein_experiments import samplers
import tensorflow as tf
import numdifftools as nd
from tools.kernels import getGaussianKernelWithDerivatives_metric
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