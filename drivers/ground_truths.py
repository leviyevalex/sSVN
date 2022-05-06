import bilby
import os
import numpy as np
import matplotlib
matplotlib.use('module://backend_interagg')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
import pickle
from scipy.stats import ks_2samp
# import mmd
import deepdish as dd
import h5py
#%%
class rosenbrock(bilby.Likelihood):
    def __init__(self):
        """
        A very simple Gaussian likelihood

        Parameters
        ----------
        data: array_like
            The data to analyse
        """
        super().__init__(parameters={'x': None, 'y': None})
        self.stdn = 0.3
        self.varn = self.stdn ** 2
        self.nData = 1
        np.random.seed(40)
        self.DoF = 2
        # self.thetaTrue = np.random.normal(size=self.DoF)
        self.thetaTrue = np.random.normal(size = self.DoF)
        self.data = self.simulateData()
        # self.data = data
        # self.N = len(data)

    def log_likelihood(self):
        x = self.parameters['x']
        y = self.parameters['y']
        theta = np.array([x, y])
        F = self.getForwardModel(theta)
        shift = self.data - F
        return -1 * 0.5 * np.sum(shift ** 2, 0) / self.varn

    def getForwardModel(self, theta):
        tmp = np.log((1 - theta[0]) ** 2 + 100 * (theta[1] - theta[0] ** 2) ** 2)
        return tmp

    def simulateData(self):
        noise = np.random.normal(scale=self.stdn, size=self.nData)
        return self.getForwardModel(self.thetaTrue) + noise

label = 'rosenbrock'
outdir = 'outdir'
contour_file_path = 'drivers/outdir/' + 'rosenbrock_proper_-5.00_5.00.h5'
with h5py.File(contour_file_path, 'r') as hf:
    X = hf["X"][:]
    Y = hf["Y"][:]
    Z = hf["Z"][:]

likelihood = rosenbrock()
names = ['x', 'y']  # set the parameter names
mu = np.array([0., 0.])  # the means of the parameters
cov = np.linalg.inv(np.array([[1, 0],
                              [0, 1]]))
mvg = bilby.core.prior.MultivariateGaussianDist(names, mus=mu, covs=cov)
priors = dict()
priors['x'] = bilby.core.prior.MultivariateGaussian(mvg, 'x')
priors['y'] = bilby.core.prior.MultivariateGaussian(mvg, 'y')
result = bilby.run_sampler(likelihood=likelihood, priors=priors, sampler='dynesty', npoints=500, walks=50, outdir=outdir, label=label)

#%%
# For running
# output_dir = os.path.dirname(os.path.abspath(__file__)) + '/outdir/'
# particle_history_path = output_dir +'rosenbrock_nP_500_300_1595598909.h5'

# For interactive
# particle_history_path = 'drivers/outdir/' + 'rosenbrock_proper_nP_3000_300_1596075815.h5'
# composite_map = dd.io.load(particle_history_path)
# final_stein_samples = composite_map[36]['X']
# plt.title('Stein vs Dynesty Samples')
x_dyn = np.copy(result._samples[:,0])
y_dyn = np.copy(result._samples[:,1])

# x_stein = np.copy(final_stein_samples[:,0])
# y_stein = np.copy(final_stein_samples[:,1])
#%%
dd.io.save('3k_samples_rosen_test.h5', np.vstack((x_dyn, y_dyn)).T)


#%%
# Double Banana Figure
fig, ax = plt.subplots()
cp = ax.contourf(X, Y, Z, 10)
ax.contourf(X, Y, Z, 10)
fig.colorbar(cp)
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_title('Double Banana')
fig.show()
#%%
# KDE Plots
# p = sns.jointplot(x_stein, y_stein, kind="kde", height=7, space=0)
# p.fig.suptitle('Stein sample KDE')
# plt.show()
p = sns.jointplot(x_dyn, y_dyn, kind="kde", height=7, space=0)
p.fig.suptitle('Dynesty sample KDE')
plt.show()
#%%
# Sample comparison
fig, ax = plt.subplots()
ax.scatter(x_dyn, y_dyn, s=.5,c='b', label='Dynesty', alpha=1)
# ax.scatter(x_stein, y_stein, s=.5, c='r', label='Stein', alpha=1)
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_title('Double Banana Samples')
ax.legend()
ax.grid(True)
ax.set_facecolor('white')
plt.grid(b=None)
# plt.axis('off')
fig.show()
#%%
a = dd.io.load('prior-sGaussian-like-doubleRosenbrock-300k_samples.h5')
