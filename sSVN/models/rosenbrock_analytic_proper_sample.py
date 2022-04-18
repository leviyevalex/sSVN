import numpy as np
from scipy.stats import truncnorm
class rosenbrock_analytic:

    def __init__(self, *arg):
        self.DoF = 2
        self.modelType = 'rosenbrock_proper'

        # Range to plot
        self.begin = -2.5
        self.end = 2.5

        self.id = 'double-banana'

        # Works for non-symmetric covariances as well!
        self.sigmaInv = np.linalg.inv(np.array([[1, 0],
                                                [0, 1]]))

        self.mu = np.array([0, 0])
        self.nData = 1
        # for the likelihood
        self.stdn = 0.3
        self.varn = self.stdn ** 2
        np.random.seed(40)
        # self.thetaTrue = np.random.normal(size=self.DoF)
        self.thetaTrue = np.random.normal(size = self.DoF)
        self.data = self._simulateData()

        self.nLikelihoodEvaluations = 0
        self.nGradLikelihoodEvaluations = 0
        self.nFisherLikelihoodEvaluations = 0

    #########################
    ### PRIORS
    #########################
    def _getMinusLogPrior(self, theta):
        temp = (theta-self.mu)
        return 1/2 * (temp).T @ self.sigmaInv @ (temp)
    def _getGradientMinusLogPrior(self, theta):
        temp = (theta-self.mu)
        return 1/2 * self.sigmaInv @ (temp) + 1/2 * (temp).T @ self.sigmaInv

    def _getHessianMinusLogPrior(self, theta):
        return self.sigmaInv

    def _newDrawFromPrior(self, nParticles):
        # return np.random.uniform(low=-3, high=3, size=(nParticles, self.DoF))
        return np.random.uniform(low=-6, high=6, size=(nParticles, self.DoF))
        # return np.random.multivariate_normal(mean=self.mu, cov=self.sigmaInv, size=nParticles)
    #########################
    ### LIKELIHOOD
    #########################
    def _getForwardModel(self, theta):
        #\reals^{d} \to \reals
        tmp = np.log((1 - theta[0]) ** 2 + 100 * (theta[1] - theta[0] ** 2) ** 2)
        return tmp

    def _getJacobianForwardModel(self, theta):
        # \reals^d \to \reals^d
        tmp = np.array([(2 * (theta[0] - 1 - 200 * theta[0] * (theta[1] - theta[0] ** 2))) / (1 + theta[0] ** 2 - 2 * theta[0] + 100 * (theta[1] - theta[0] ** 2) ** 2), (200 * (theta[1] - theta[0] ** 2)) / (1 + theta[0] ** 2 - 2 * theta[0] + 100 * (theta[1] - theta[0] ** 2) ** 2)])
        return tmp

    def _getHessForwardModel(self, theta):
        x = theta[0]
        y = theta[1]
        tmp = np.array([[(1200 * x ** 2 - 400 * y + 2) / ((1 - x) ** 2 + 100 * (-x ** 2 + y) ** 2) - (-2 + 2 * x - 400 * (-x ** 2 + y) * x) ** 2 / ((1 - x) ** 2 + 100 * (-x ** 2 + y) ** 2) ** 2,-400 * x / ((1 - x) ** 2 + 100 * (-x ** 2 + y) ** 2) - (-2 + 2 * x - 400 * (-x ** 2 + y) * x) / ((1 - x) ** 2 + 100 * (-x ** 2 + y) ** 2) ** 2 * (-200 * x ** 2 + 200 * y)],[-400 * x / ((1 - x) ** 2 + 100 * (-x ** 2 + y) ** 2) - (-2 + 2 * x - 400 * (-x ** 2 + y) * x) / ((1 - x) ** 2 + 100 * (-x ** 2 + y) ** 2) ** 2 * (-200 * x ** 2 + 200 * y),200 / ((1 - x) ** 2 + 100 * (-x ** 2 + y) ** 2) - (-200 * x ** 2 + 200 * y) ** 2 / ((1 - x) ** 2 + 100 * (-x ** 2 + y) ** 2) ** 2]])
        return tmp

    def _simulateData(self):
        noise = np.random.normal(scale=self.stdn, size=self.nData)
        return self._getForwardModel(self.thetaTrue) + noise

    def getMinusLogLikelihood(self, theta):
        # Recall that F is a scalar
        F = self._getForwardModel(theta)
        shift = self.data - F
        tmp = 0.5 * np.sum(shift ** 2, 0) / self.varn
        self.nLikelihoodEvaluations += 1
        return tmp

    def getGradientMinusLogLikelihood(self, theta):
        F = self._getForwardModel(theta)
        J = self._getJacobianForwardModel(theta)
        tmp = -1 * J * np.sum(self.data - F, 0) / self.varn
        self.nGradLikelihoodEvaluations += 1
        return tmp

    def getGNHessianMinusLogLikelihood(self, theta):
        J = self._getJacobianForwardModel(theta)
        self.nFisherLikelihoodEvaluations += 1
        # return np.einsum('i,j -> ij', J, J) / self.varn
        return np.outer(J, J) / self.varn

    def getHessianMinusLogLikelihood(self, theta):
        F = self._getForwardModel(theta)
        J = self._getJacobianForwardModel(theta)
        H = self._getHessForwardModel(theta)
        shift = self.data - F
        return (self.nData * np.outer(J, J) - H * np.sum(shift, 0)) / self.varn

    def getGradientGNHessianMinusLogLikelihood(self, theta):
        J = self._getJacobianForwardModel(theta)
        HF = self._getHessForwardModel(theta)
        return (np.einsum('jk,i -> ijk', HF, J) + np.einsum('j,ik -> ijk', J, HF)) / self.varn

    def getMinusLogPosterior(self, theta):
        return self.getMinusLogLikelihood(theta) + self._getMinusLogPrior(theta)

    def getGradientMinusLogPosterior(self, theta):
        return self.getGradientMinusLogLikelihood(theta) + self._getGradientMinusLogPrior(theta)

    def getGNHessianMinusLogPosterior(self, theta):
        return self.getGNHessianMinusLogLikelihood(theta) + self._getHessianMinusLogPrior(theta)

    def getHessianMinusLogPosterior(self, theta):
        return self.getHessianMinusLogLikelihood(theta) + self._getHessianMinusLogPrior(theta)

    # Vectorized versions of previous methods
    # Args:
    #   thetas (array): N x DoF array, where N is number of samples
    # Returns: (array) N x 1, N x DoF, N x (DoF x DoF), N x (DoF x DoF) respectively

    def getMinusLogPosterior_ensemble(self, thetas):
        return np.apply_along_axis(self.getMinusLogPosterior, 1, thetas)

    def getGradientMinusLogPosterior_ensemble(self, thetas):
        return np.apply_along_axis(self.getGradientMinusLogPosterior, 1, thetas)

    def getGNHessianMinusLogPosterior_ensemble(self, thetas):
        return np.apply_along_axis(self.getGNHessianMinusLogPosterior, 1, thetas)

    def getHessianMinusLogPosterior_ensemble(self, thetas):
        return np.apply_along_axis(self.getHessianMinusLogPosterior, 1, thetas)