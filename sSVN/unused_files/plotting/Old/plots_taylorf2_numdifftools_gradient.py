import numpy as np
import matplotlib.pyplot as plt
from source.stein_experiments import BILBY_TAYLORF2
from mpl_toolkits import mplot3d
from matplotlib import cm
import h5py
# %%
# Choose a point to inject
inject = np.array([2.0, 2.0])

# Create model
bilby_model = BILBY_TAYLORF2(inject)

# Choose grid length
ngrid = 3 # Default 100 works nicely

#fig = plt.figure()
ax = plt.axes(projection='3d')

# Create mesh (make sure to make entries floats, otherwise plots come out wrong.
m1 = np.linspace(1.99, 2.01, ngrid)
m2 = np.linspace(1.99, 2.01, ngrid)
M1, M2 = np.meshgrid(m1, m2)

# %%
gradientLikelihood0 = bilby_model.getGradientMinusLogLikelihood(np.vstack((np.ndarray.flatten(M1), np.ndarray.flatten(M2))))[0].reshape(ngrid, ngrid)
#gradientLikelihood1 = bilby_model.getGradientMinusLogLikelihood(np.vstack((np.ndarray.flatten(M1), np.ndarray.flatten(M2))))[1].reshape(ngrid, ngrid)

# %%
# Create surface heat map

fig1 = plt.figure(figsize = (15, 15))
ax1 = fig1.add_subplot(1, 1, 1, projection='3d')
#ax1 = plt.axes(projection = '3d')
plt.rc('font', family='serif')
plt.locator_params(nbins=8)
ax1.plot_surface(M1, M2, gradientLikelihood0, rstride = 3, cstride = 3, cmap=cm.coolwarm)
ax1.set_xlabel('mass_1')
ax1.set_ylabel('mass_2')
ax1.set_title('\nabla \log \mathscr{L}')
fig1.savefig('gradient0_likelihood_surface.png')


