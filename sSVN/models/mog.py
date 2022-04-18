import numpy as np
class mog:
    def __init__(self, *arg):
        # Make sure its symmetric

        # self.sigma_p = np.array([[1/16, 0],
        #                          [0, 1/16]])
        self.sigma = np.array([[10, 0],
                               [0, 10]])
        # self.sigma_l = np.array([[1, 0],
        #                          [0, 1]])

        self.sigmaInv = np.linalg.inv(self.sigma)

        # self.sigmaInv_l = np.linalg.inv(self.sigma_l)

        self.mu = np.array([0, 0])
        # self.mu_p = np.array([3, 3])
        # self.mu_p = np.array([0, 0])
        self.DoF = self.mu.size
        self.modelType = 'mog'

        # Range to plot
        self.begin = -5
        self.end = 5

        self.nLikelihoodEvaluations = 0
        self.nGradLikelihoodEvaluations = 0
        self.nFisherLikelihoodEvaluations = 0
        self.centers = np.array([[-2, -2],
                                 [-2, 0],
                                 [-2, 2],
                                 [0,-2],
                                 [0,0],
                                 [0,2],
                                 [2,-2],
                                 [2, 0],
                                 [2, 2]])
    #########################
    ### PRIORS
    #########################
    def getMinusLogPrior_individual(self, theta):
        # temp = (theta-self.mu_p)
        # return 1/2 * (temp).T @ self.sigmaInv_p @ (temp)
        return 0
    def getGradientMinusLogPrior_individual(self, theta):
        # temp = (theta-self.mu_p)
        # return 1/2 * self.sigmaInv_p @ (temp) + 1/2 * (temp).T @ self.sigmaInv_p
        return 0

    def getGNHessianMinusLogPrior_individual(self, theta):
        # return self.sigmaInv_p
        return 0

    def newDrawFromPrior(self, nParticles):
        return np.random.multivariate_normal(mean=self.mu, cov=self.sigma, size=nParticles)
        # return 3. + 0.25 * np.random.normal(0, 1, (nParticles, self.DoF))
    #########################
    ### LIKELIHOOD
    #########################
    # 1
    def getMinusLogLikelihood_individual(self, theta):
        tmp1 = 0
        for l in range(9):
            temp = (theta-self.centers[l])
            tmp1 += 1/2 * (temp).T @ self.sigmaInv @ (temp)
        return tmp1

    def getGradientMinusLogLikelihood_individual(self, theta):
        tmp1 = np.zeros(self.DoF)
        for l in range(9):
            temp = (theta-self.centers[l])
            tmp1 += 1/2 * self.sigmaInv @ (temp) + 1/2 * (temp).T @ self.sigmaInv
        return tmp1

    def getGNHessianMinusLogLikelihood_individual(self,theta):
        return self.sigmaInv * 9

    def getGradientGNHessianMinusLogLikelihood_individual(self, theta):
        return np.zeros((self.DoF, self.DoF, self.DoF))


