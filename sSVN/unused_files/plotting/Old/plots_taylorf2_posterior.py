import numpy as np
import matplotlib.pyplot as plt
from source.stein_experiments import BILBY_TAYLORF2, SVN
from mpl_toolkits import mplot3d
from matplotlib import cm
import h5py
# %%
# Choose a point to inject
inject = np.array([10, 10])

# Create model
bilby_model = BILBY_TAYLORF2(inject)
SVN = SVN()

# Choose grid length
ngrid = 100 # Default 100 works nicely

#fig = plt.figure()
ax = plt.axes(projection='3d')

# Create mesh (make sure to make entries floats, otherwise plots come out wrong.
m1 = np.linspace(-1, 5, ngrid)
m2 = np.linspace(-1, 5, ngrid)
M1, M2 = np.meshgrid(m1, m2)

Posterior = bilby_model.getMinusLogPosterior(np.vstack((np.ndarray.flatten(M1), np.ndarray.flatten(M2)))).reshape(ngrid, ngrid)

# %%
# Create surface heat map
fig1 = plt.figure(figsize = (15, 15))
ax1 = plt.axes(projection = '3d')
plt.rc('font', family='serif')
plt.locator_params(nbins=8)
ax1.plot_surface(M1, M2, Posterior, rstride = 3, cstride = 3, cmap=cm.coolwarm)
ax1.set_xlabel('mass_1')
ax1.set_ylabel('mass_2')
ax1.set_zlim(-100, 20000)
ax1.set_title('Log Posterior + boundary')
fig1.savefig('posterior_boundary.png')

# %%
# Create contour plot
fig2 = plt.figure(figsize = (15, 15))
plt.rc('font', family='serif')
plt.locator_params(nbins=8)
ax2 = plt.axes()
ax2.contourf(M1, M2, Posterior, 100)
ax2.set_xlabel('mass_1')
ax2.set_ylabel('mass_2')
ax2.set_title('Normalized likelihood')
fig2.savefig('contour_plot_of_posterior.png')
#%%
# Create line plot
direction = np.array([np.sqrt(2), np.sqrt(2)])
begin = np.array([8, 8])
.phi_line_plot(beginning_vector=begin, direction=direction, length_of_line=5)
