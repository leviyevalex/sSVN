import numpy as np
import matplotlib.pyplot as plt
from source.stein_experiments import BILBY_TAYLORF2
from mpl_toolkits import mplot3d
from matplotlib import cm
import h5py
# %%
inject = np.array([10.0, 10.0])
model = BILBY_TAYLORF2(inject)
# %%
ngrid = 100
# Create mesh (make sure to make entries floats, otherwise plots come out wrong.
m1 = np.linspace(9, 11, ngrid)
m2 = np.linspace(9, 11, ngrid)
M1, M2 = np.meshgrid(m1, m2)

# %%
prior_z = model.getMinusLogPrior(np.vstack((np.ndarray.flatten(M1), np.ndarray.flatten(M2)))).reshape(ngrid, ngrid)

# %%
# Create surface heat map

fig1 = plt.figure(figsize = (15, 15))
ax1 = fig1.add_subplot(1, 1, 1, projection='3d')
#ax1 = plt.axes(projection = '3d')
plt.rc('font', family='serif')
plt.locator_params(nbins=8)
ax1.plot_surface(M1, M2, prior_z, rstride = 3, cstride = 3, cmap=cm.coolwarm)
ax1.set_xlabel('mass_1')
ax1.set_ylabel('mass_2')
ax1.set_title('Priors')
fig1.savefig('m1_m2_priors.png')
