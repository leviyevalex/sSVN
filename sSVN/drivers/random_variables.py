import scipy
import numpy as np
import matplotlib.pyplot as plt
#%%
mean = np.array([0, 0])
cov = np.array([[1, 2],
                [2, 8]])
a = np.random.multivariate_normal(mean, cov, 5) # returns n x d as desired.
#%%
nsamples = 1000
samples = np.random.multivariate_normal(mean, cov, nsamples)
x = samples[:, 0]
y = samples[:, 1]
#%%
# Modify each sample with K and see how they're distributed
K = np.array([[3, 1.8],
              [1.5, 5]])
modified_samples = (np.einsum('db,mb -> dm', K, samples).flatten()).reshape(nsamples, 2, order='F')
for n in range(nsamples):
    assert np.allclose(modified_samples[n], K @ samples[n], rtol=1e-6)
x_mod = modified_samples[:, 0]
y_mod = modified_samples[:, 1]
#%%
# Which modified noise is correct?
# samples_test = np.random.multivariate_normal(mean, K.T @ cov @ K, nsamples)
samples_test = np.random.multivariate_normal(mean, K @ cov @ K.T, nsamples) # LOOKS LIKE THIS WORKS
x_test = samples_test[:, 0]
y_test = samples_test[:, 1]
plt.scatter(x_test, y_test)
plt.scatter(x_mod, y_mod)
# plt.scatter(x, y)
plt.show()