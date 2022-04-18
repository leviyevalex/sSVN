import numpy as np
from scipy.stats import truncnorm
class rosenbrock_analytic:
    def __init__(self, *arg):

        self.DoF = 2
        self.modelType = 'rosenbrock'

        # Range to plot
        self.begin = -5
        self.end = 5

        # Range to sample from
        # self.low = -3
        # self.high = 3

        # Works for non-symmetric covariances as well!
        self.sigmaInv = np.linalg.inv(np.array([[0.5, 0],
                                                [0, 0.5]]))

        self.mu = np.array([1, 1])

    # 1
    def getMinusLogLikelihood_individual(self, theta):
        x = theta[0]
        y = theta[1]

        return np.log((1 - x) ** 2 + 100 * (-x ** 2 + y) ** 2)

    # 4
    def getGradientMinusLogLikelihood_individual(self, theta):
        x = theta[0]
        y = theta[1]

        return np.array([(-2 + 2 * x - 400 * (-x ** 2 + y) * x) / ((1 - x) ** 2 + 100 * (-x ** 2 + y) ** 2),
                           (-200 * x ** 2 + 200 * y) / ((1 - x) ** 2 + 100 * (-x ** 2 + y) ** 2)])


    # 6
    def getGNHessianMinusLogLikelihood_individual(self,theta):
        x = theta[0]
        y = theta[1]
        # Full hessian
        # return np.array([[(1200 * x ** 2 - 400 * y + 2) / ((1 - x) ** 2 + 100 * (-x ** 2 + y) ** 2) - (
        #             -2 + 2 * x - 400 * (-x ** 2 + y) * x) ** 2 / ((1 - x) ** 2 + 100 * (-x ** 2 + y) ** 2) ** 2,
        #                       -400 * x / ((1 - x) ** 2 + 100 * (-x ** 2 + y) ** 2) - (
        #                                   -2 + 2 * x - 400 * (-x ** 2 + y) * x) / (
        #                                   (1 - x) ** 2 + 100 * (-x ** 2 + y) ** 2) ** 2 * (-200 * x ** 2 + 200 * y)], [
        #                          -400 * x / ((1 - x) ** 2 + 100 * (-x ** 2 + y) ** 2) - (
        #                                      -2 + 2 * x - 400 * (-x ** 2 + y) * x) / (
        #                                      (1 - x) ** 2 + 100 * (-x ** 2 + y) ** 2) ** 2 * (-200 * x ** 2 + 200 * y),
        #                          200 / ((1 - x) ** 2 + 100 * (-x ** 2 + y) ** 2) - (-200 * x ** 2 + 200 * y) ** 2 / (
        #                                      (1 - x) ** 2 + 100 * (-x ** 2 + y) ** 2) ** 2]])

        return np.array([[(-2 + 2 * x - 400 * (-x ** 2 + y) * x) ** 2 / (
                    (1 - x) ** 2 + 100 * (-x ** 2 + y) ** 2) ** 2,
                                (-2 + 2 * x - 400 * (-x ** 2 + y) * x) / ((1 - x) ** 2 + 100 * (-x ** 2 + y) ** 2) ** 2 * (
                                            -200 * x ** 2 + 200 * y)], [(-2 + 2 * x - 400 * (-x ** 2 + y) * x) / (
                    (1 - x) ** 2 + 100 * (-x ** 2 + y) ** 2) ** 2 * (-200 * x ** 2 + 200 * y),
                                                                        (-200 * x ** 2 + 200 * y) ** 2 / (
                                                                                    (1 - x) ** 2 + 100 * (
                                                                                        -x ** 2 + y) ** 2) ** 2]])

    def getMinusLogPrior_individual(self, theta):
        # term = 1/2 * (theta - self.mu).T @ self.SigmaInv @ (theta - self.mu)
        # assert(term.type)
        temp = (theta-self.mu)
        return 1/2 * (temp).T @ self.sigmaInv @ (temp)
    def getGradientMinusLogPrior_individual(self, theta):
        temp = (theta-self.mu)
        return 1/2 * self.sigmaInv @ (temp) + 1/2 * (temp).T @ self.sigmaInv
    def getGNHessianMinusLogPrior_individual(self, theta):
        return self.sigmaInv

    def newDrawFromPrior(self, nParticles):
        return np.random.multivariate_normal(mean=self.mu, cov=self.sigmaInv, size=nParticles)

        # def get_truncated_normal(self, mu, sigma, low, high):
        #     return truncnorm(
        #         (low - mu) / sigma, (high - mu) / sigma, loc=mu, scale=sigma)
        # return np.random.normal(loc = self.mu[0], scale=self.sigma, size=(self.DoF,nParticles))
    # def newDrawFromPrior(self, nParticles):
    #     # Fix mu later
    #     trunc_norm = self.get_truncated_normal(self.mu[0], self.sigma, self.low, self.high)
    #     particles = np.zeros((nParticles, self.DoF))
    #     for n in range(nParticles):
    #         particles[n] = trunc_norm.rvs(self.DoF)
    #     return particles.T
    # # 2
    # def getMinusLogPrior_individual(self, theta):
    #     return 1
    # # 3
    # def getGradientMinusLogPrior_individual(self, theta):
    #     return 0
    # # 5
    # def getGNHessianMinusLogPrior_individual(self,theta):
    #     return 0
    # # 7
    # def newDrawFromPrior(self, nParticles):
    #     np.random.seed(1)
    #     particles = np.random.uniform(low = self.low, high = self.high, size = self.DoF).reshape(self.DoF, 1)
    #     while particles.shape[1] != nParticles:
    #         particles = np.hstack((particles, (np.random.uniform(low=self.low, high=self.high, size=self.DoF)).reshape(self.DoF, 1)))
    #     return particles
    #