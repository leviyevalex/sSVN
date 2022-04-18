# import numpy as np
from scipy.stats import truncnorm
import autograd.numpy as np
# import autograd as ag
from autograd import grad, hessian, jacobian
from numdifftools import Gradient
class hybrid_rosenbrock:

    def __init__(self, *arg):
        # self.modelType = 'hybrid_rosenbrock'
        # Range to plot
        self.begin = -2.5
        self.end = 2.5
        self.n1i = 2
        self.n2j = 1
        self.modelType = 'hybrid_rosenbrock_%i_%i' % (self.n1i, self.n2j)
        # self.n1i = 3
        # self.n2j = 2

        self.DoF = self.getDimension(self.n1i, self.n2j)
        # self.DoF = (self.n1i - 1) * self.n2j + 1
        # self.bji = np.ones((self.n2j, self.n1i)) * (100 / 20)
        self.bji = np.ones(self.DoF) * (100 / 20)
        self.mu = 1
        self.a = 1 / 20

        # Works for non-symmetric covariances as well!
        # self.sigmaInv_gauss = np.linalg.inv(np.array([[1, 0],
        #                                               [0, 1]]))

        self.sigmaInv_gauss = np.eye(self.DoF)
        self.mu_gauss = np.zeros(self.DoF)
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

    def getGNHessianMinusLogPrior_individual(self, theta):
        return 0

    def _newDrawFromPrior(self, nParticles):
        return np.random.multivariate_normal(mean=self.mu_gauss, cov=self.sigmaInv_gauss, size=nParticles)
    #########################
    ### LIKELIHOOD
    #########################
    def getDimension(self, n1, n2):
        return (n1 - 1) * n2 + 1

    def getMinusLogLikelihood_individual_2d(self, theta):
        bji = np.ones((self.n2j, self.n1i)) * (100 / 20)
        # assert(theta.size == self.DoF)
        # the i and j have to be subtracted by 1!!!
        temp = self.a * (theta[0] - self.mu) ** 2 + bji[0, 1] * (theta[1] - theta[0] ** 2) ** 2
        return temp

    def getMinusLogLikelihood_individual(self, theta):
        theta = theta.astype('float64')
        sum = 0
        for j in range(self.n2j):
            for i in range(1, self.n1i):
                sum += self.bji[self.ji_to_index(j, i)] * (theta[self.ji_to_index(j, i)] - theta[self.ji_to_index(j, i - 1)] ** 2) ** 2
        mlnhr = self.a * (theta[0] - self.mu) ** 2 + sum
        self.nLikelihoodEvaluations += 1
        return mlnhr

    def ji_to_index(self, j, i):
        if i == 0:
            return 0
        else:
            return j * self.n2j + i

    def getMinusLogLikelihood_individual_5d(self, theta):
        x1 = theta[0]
        x12 = theta[1]
        x13 = theta[2]
        x22 = theta[3]
        x23 = theta[4]
        mu=1
        a=1/20
        b13 = 100/20
        b12 = 100/20
        b22 = 100/20
        b23 = 100/20
        self.nLikelihoodEvaluations += 1
        return np.exp(-a * (x1 - mu) ** 2 - b12 * (x12 - x1 ** 2) ** 2 - b13 * (x13 - x12 ** 2) ** 2
                      - b22 * (x22 - x1 ** 2) ** 2 - b23 * (x23 - x22 ** 2) ** 2)

    def getGradientMinusLogLikelihood_individual(self, theta):
        # ALGORITHMIC DIFFERENTIATION
        theta = theta.astype('float64')
        self.nGradLikelihoodEvaluations += 1
        return jacobian(self.getMinusLogLikelihood_individual)(theta)


    def getGNHessianMinusLogLikelihood_individual(self, theta):
        J = jacobian(self.getMinusLogLikelihood_individual)(theta)
        self.nFisherLikelihoodEvaluations += 1
        return np.outer(J, J)

    # def getMinusLogLikelihood_individual_general(self, theta):
    #     x_block = np.zeros((self.n2j, self.n1i))
    #     x_block[:, 0] = theta[0]
    #     x_block[:, 1:self.DoF] = theta[1:self.DoF].reshape(self.n2j, self.n1i - 1)
    #     sum = 0
    #     for j in range(self.n2j):
    #         for i in range(1, self.n1i):
    #             sum += self.bji[j, i] * (x_block[j, i] - x_block[j, i - 1] ** 2) ** 2
    #     hr = np.exp(-self.a * (x_block[0, 0] - self.mu) ** 2 - sum)
    #     # self.a * (x_block[0, 0] + self.mu) ** 2 + sum)
    #     return -np.log(hr)
    # def getMinusLogLikelihood_individual(self, theta):
    #     theta = theta.astype('float64')
    #     x_block = np.zeros((self.n2j, self.n1i))
    #     # x_block = x_block.astype('float64')
    #     # x_block[:, 0] = np.ones(self.n2j) * theta[0]
    #     x_block[:, 0] = theta[0]
    #     x_block[:, 1:self.DoF] = theta[1:self.DoF].reshape(self.n2j, self.n1i - 1)
    #     sum = 0
    #     for j in range(self.n2j):
    #         for i in range(1, self.n1i):
    #             sum += self.bji[j, i] * (x_block[j, i] - x_block[j, i - 1] ** 2) ** 2
    #             # sum += self.bji[j-1, i-1] * (x_block[j-1, i-1] - x_block[j-1, i - 2] ** 2) ** 2
    #     mlnhr = self.a * (x_block[0, 0] - self.mu) ** 2 + sum
    #     # self.a * (x_block[0, 0] + self.mu) ** 2 + sum)
    #     return mlnhr




def main():
    hr = hybrid_rosenbrock()
    array = np.array([1., 2.])
    res1 = hr.getMinusLogLikelihood_individual(array)
    res2 = hr.getMinusLogLikelihood_individual_2d(array)
    grad = hr.getGradientMinusLogLikelihood_individual(array)
    # res2 = hr.getMinusLogLikelihood_individual_naive(array)
    dum = 1 + 1
if __name__ is '__main__':
    main()