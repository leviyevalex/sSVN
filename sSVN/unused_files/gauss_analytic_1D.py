import numpy as np
class gauss_analytic_1D:
    def __init__(self, *arg):
        self.SigmaInv = np.array([1])
        self.mu = 0
        self.DoF = 1
        self.modelType = 'gauss_analytic_1D'

        # Range to plot
        self.begin = -5
        self.end = 5

        # Range to sample from
        self.low = -3
        self.high = 3

    # 1
    def getMinusLogLikelihood_individual(self, theta):
        theta = theta.reshape(self.DoF, 1)
        return 1/2 * (theta - self.mu).T @ self.SigmaInv @ (theta - self.mu)
    # 2
    def getMinusLogPrior_individual(self, theta):
        theta = theta.reshape(self.DoF, 1)
        return 1
    # 3
    def getGradientMinusLogPrior_individual(self, theta):
        theta = theta.reshape(self.DoF, 1)
        return 0
    # 4
    def getGradientMinusLogLikelihood_individual(self, theta):
        theta = theta.reshape(self.DoF, 1)
        return self.SigmaInv @ (theta - self.mu)
    # 5
    def getGNHessianMinusLogPrior_individual(self,theta):
        theta = theta.reshape(self.DoF, 1)
        return 0
    # 6
    def getGNHessianMinusLogLikelihood_individual(self,theta):
        theta = theta.reshape(self.DoF, 1)
        return self.SigmaInv
    # 7
    def newDrawFromPrior(self, nParticles):
        np.random.seed(1)

        particles = np.random.uniform(low = self.low, high = self.high, size = self.DoF).reshape(self.DoF, 1)
        while particles.shape[1] != nParticles:
            particles = np.hstack((particles, (np.random.uniform(low=self.low, high=self.high, size=self.DoF)).reshape(self.DoF, 1)))
        return particles


