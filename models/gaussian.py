import numpy as np
class gauss_analytic:
    def __init__(self, *arg):
        # Make sure its symmetric

        # self.sigma_p = np.array([[1/16, 0],
        #                          [0, 1/16]])
        # self.sigma_p = np.array([[1, 0],
        #                          [0, 1]])

        # Precision Matrix
        self.sigma_l = np.array([[1, 0],
                                 [0, 1]])

        # self.sigmaInv_p = np.linalg.inv(self.sigma_p)

        self.sigmaInv_l = np.linalg.inv(self.sigma_l)

        self.mu_l = np.array([0, 0])
        # self.mu_p = np.array([3, 3])
        self.mu_p = np.array([0, 0])
        self.DoF = self.mu_l.size
        self.modelType = 'gauss_analytic'

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
        return np.random.multivariate_normal(mean=self.mu_p, cov=self.sigma_p, size=nParticles)
        # return 3. + 0.25 * np.random.normal(0, 1, (nParticles, self.DoF))
    #########################
    ### LIKELIHOOD
    #########################
    # 1
    def getMinusLogLikelihood_individual(self, theta):
        temp = (theta-self.mu_l)
        return 1/2 * (temp).T @ self.sigmaInv_l @ (temp)

    def getGradientMinusLogLikelihood_individual(self, theta):
        temp = (theta-self.mu_l)
        return 1/2 * self.sigmaInv_l @ (temp) + 1/2 * (temp).T @ self.sigmaInv_l

    def getGNHessianMinusLogLikelihood_individual(self,theta):
        return self.sigmaInv_l

    def getGradientGNHessianMinusLogLikelihood_individual(self, theta):
        return np.zeros((self.DoF, self.DoF, self.DoF))


