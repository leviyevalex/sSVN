# import numpy as np
import autograd.numpy as np
from autograd.scipy import stats
from autograd import hessian, jacobian, grad
from scipy.stats import multivariate_normal as mvn
class gauss_stats:
    def __init__(self, DoF, *arg):
        self.DoF = DoF
        # self.mu = np.zeros(self.DoF)
        self.mu = np.ones(self.DoF)
        self.cov = np.eye(self.DoF)
        self.modelType = 'gauss_stats'

        # Range to plot
        self.begin = -5
        self.end = 5

        self.nLikelihoodEvaluations = 0
        self.nGradLikelihoodEvaluations = 0
        self.nFisherLikelihoodEvaluations = 0

    #########################
    ### PRIORS
    #########################
    def getMinusLogPrior_individual(self, theta):
        return 0
    def getGradientMinusLogPrior_individual(self, theta):
        return np.zeros(self.DoF)
    def getGNHessianMinusLogPrior_individual(self, theta):
        return np.zeros((self.DoF, self.DoF))
    def newDrawFromPrior(self, nParticles):
        # return np.random.uniform(-5, 5, size=(nParticles, self.DoF))
        # z = np.zeros(self.DoF)
        # z[1] = 5
        # return mvn.rvs(-5 * np.ones(self.DoF), self.cov, nParticles)
        return np.random.uniform(low=-6, high=6, size=(nParticles, self.DoF))
        # return mvn.rvs(z, self.cov, nParticles)
        # return stats.multivariate_normal(-5, 5, size=(nParticles, self.DoF))
    #########################
    ### LIKELIHOOD
    #########################
    # 1
    def getMinusLogLikelihood_individual(self, theta):
        return - stats.multivariate_normal.logpdf(theta, self.mu, self.cov)
    def getGradientMinusLogLikelihood_individual(self, theta):
        return jacobian(self.getMinusLogLikelihood_individual)(theta)
    def getGNHessianMinusLogLikelihood_individual(self,theta):
        return self.cov
        # return hessian(self.getMinusLogLikelihood_individual)(theta)
    def getGradientGNHessianMinusLogLikelihood_individual(self, theta):
        return np.zeros((self.DoF, self.DoF, self.DoF))
    def newDrawFromLikelihood(self, nParticles):
        return mvn.rvs(self.mu, self.cov, nParticles)
def main():
    DoF = 2
    model = gauss_stats(DoF)
    x = np.zeros(DoF)
    mlogl = model.getMinusLogLikelihood_individual(x)
    mglogl = model.getGradientMinusLogLikelihood_individual(x)
    mhlogl = model.getGNHessianMinusLogLikelihood_individual(x)
    rvs = model.newDrawFromLikelihood(5)
    pass
if __name__ is '__main__':
    main()

