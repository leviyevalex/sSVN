# import numpy as np
import autograd.numpy as np
from scipy.stats import truncnorm
from autograd import grad, hessian
import numdifftools as nd
class bimodal:

    def __init__(self, *arg):
        self.DoF = 2
        self.modelType = 'bimodal'

        # Range to plot
        self.begin = -5
        self.end = 5

        # Works for non-symmetric covariances as well!
        self.sigmaInv = np.linalg.inv(np.array([[1, 0],
                                                [0, 1]]))
        self.mu = np.array([0, 0])

        np.random.seed(40)
        self.nLikelihoodEvaluations = 0
        self.nGradLikelihoodEvaluations = 0
        self.nFisherLikelihoodEvaluations = 0

    #########################
    ### PRIORS
    #########################
    def getMinusLogPrior_individual(self, theta):
        return 0
    def getGradientMinusLogPrior_individual(self, theta):
        return 0

    def newDrawFromPrior(self, nParticles):
        return np.random.multivariate_normal(mean=self.mu, cov=self.sigmaInv, size=nParticles)

    #########################
    ### LIKELIHOOD
    #########################

    def getMinusLogLikelihood_individual(self, x):
        eplus = np.exp(-2 * (x[0] + 3) ** 2)
        eminus = np.exp(-2 * (x[0] - 3) ** 2)
        nx = np.linalg.norm(x)
        pre = np.exp(-2 * (nx - 3) ** 2)
        tmp = pre * (eplus + eminus)
        # return -tmp
        self.nLikelihoodEvaluations +=1
        return -1 * np.log(tmp)

    def getGradientMinusLogLikelihood_individual(self, theta):
        theta = theta.astype('float64')
        # return nd.Gradient(self.getMinusLogLikelihood_individual)(theta)
        self.nGradLikelihoodEvaluations +=1
        return grad(self.getMinusLogLikelihood_individual)(theta)

    def getGNHessianMinusLogLikelihood_individual(self, theta):
        J = jacobian(self.getMinusLogLikelihood_individual)(theta)
        self.nFisherLikelihoodEvaluations += 1
        return np.outer(J, J)

