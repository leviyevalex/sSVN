#%%
import numpy as np
import matplotlib.pyplot as plt
from source.stein_experiments import GAUSSIAN, SVN
import h5py
# %%
gauss_model = GAUSSIAN()

# %%
svn = SVN(gauss_model)
svn.nIterations = 100 # Converges reasonably in this amount of iterations
original_particles_x = np.copy(svn.particles[0])
original_particles_y = np.copy(svn.particles[1])
# %%
svn.apply()
# %%
ngrid = 100
x = np.linspace(-3.5, 3.5, ngrid)
y = np.linspace(-3.5, 3.5, ngrid)
X, Y = np.meshgrid(x,y)
Z = np.exp( - gauss_model.getMinusLogPosterior( np.vstack( (np.ndarray.flatten(X), np.ndarray.flatten(Y)) ) ) ).reshape(ngrid, ngrid)

# %%
# Before and after scatterplot on top of contours
fig, ax = plt.subplots(figsize = (10, 10))
plt.rc('font', family='serif')
ax.contourf(X, Y, Z, 10)
ax.scatter(svn.particles[0], svn.particles[1], marker = "x", color = 'r', label = 'After')
ax.scatter(original_particles_x, original_particles_y, marker = '.', color = 'y', label = 'Before')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Gaussian particle flow', fontdict={'fontsize': 20})
ax.legend(loc="lower right", fontsize = "x-large")
fig.savefig('gauss_particles_MAP_eliminated.png')

