#%%
import numpy as np
import h5py
import os
from source.stein_experiments import SVI
from models.rosenbrock_analytic_proper_sample import rosenbrock_analytic
import matplotlib.pyplot as plt
import copy
import deepdish as dd
#%%
model = rosenbrock_analytic()
stein = SVI(model, nIterations=100, nParticles=500, optimizeMethod='SVGD')
nParticlesT = 15000
T = model.newDrawFromPrior(nParticlesT)
T1 = copy.deepcopy(T)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file = 'output_data_new.h5'
#%%
pushed_samples = stein.pushForward(samples = T, history_path=file)
#%%
# Load contour file and particles for map
cwd = os.getcwd()
background_path = os.path.join(cwd, 'drivers/outdir', 'rosenbrock_proper_-2.50_2.50.h5')
background_data = dd.io.load(background_path)
X = background_data['X']
Y = background_data['Y']
Z = background_data['Z']
with h5py.File(file, 'r') as f:
    X0 = f['0']['X'][()]
    X_final = f['final_updated_particles']['X'][()]
    num_iter = f['metadata'].attrs.__getitem__('total_num_iterations')
    x_iter = f['%i' % (num_iter - 1)]['X'][()]
#%%
test_push = stein.pushForward(X0, file)
#%%
# Figure settings
plt.figure(dpi=1200)
plt.rc('font', family='serif')
fig, ax = plt.subplots(1, 2, figsize=(15,7))
ax[0].axis('off')
ax[1].axis('off')
# Plotting
ax[0].scatter(T[:,0], T[:, 1], marker=".", color='grey', alpha=0.5, s=3)
ax[0].scatter(X0[:,0], X0[:, 1], marker=".", color='#51AEFF', s=8)
ax[1].scatter(pushed_samples[:,0], pushed_samples[:, 1], marker=".", color='grey', alpha=0.5, s=3, label='Test set')
ax[1].scatter(X_final[:,0], X_final[:, 1], marker=".", color='#51AEFF', s=8, label='Basis set')
ax[1].legend(loc='upper right')
fig.savefig('holes_in_map.pdf')
#%%
fig.savefig('holes_in_map.png', dpi=1500)
#%%
fig.show()












# fig, ax = plt.subplots(2, 1, figsize = (10, 10))
# #%%
# ax[0, 0].contour(X, Y, Z, 7, colors='black', alpha=0.1)
# ax[1, 0].contour(X, Y, Z, 7, colors='black', alpha=0.1)
# # ax[0, 1].contour(X, Y, Z, 7, colors='black', alpha=0.1)
# # ax[1, 1].contour(X, Y, Z, 7, colors='black', alpha=0.1)
# #%%
# # ax.set_facecolor('#F5FEFF')
# # fig.patch.set_facecolor('#F5FEFF')
# ax[0, 0].axis('off')
# # ax[1, 1].axis('off')
# ax[1, 0].axis('off')
# # ax[0, 1].axis('off')
#
# ax[0, 0].scatter(pushed_samples[:,0], pushed_samples[:, 1], marker=".", color='#51AEFF', s=8, label='Final test set')
# ax[1, 0].scatter(T[:,0], T[:, 1], marker="x", s=8, label='Initial test set')
# ax[1, 0].scatter(X0[:,0], X0[:, 1], marker="x", color='r', s=8, label='Initial basis set')
# ax[0, 1].scatter(X_final[:,0], X_final[:, 1], marker="x", s=8, label='Final basis set')
# #%%
# ax[0, 0].legend(loc='upper right')
# ax[1, 0].legend(loc='upper right')
# ax[1, 1].legend(loc='upper right')
# ax[0, 1].legend(loc='upper right')
# ax[0, 1].scatter(pushed_samples[:,0], pushed_samples[:, 1], marker="x", s=8, label='Pushed inportance samples')
#%%
# plt.legend()
fig.show()
# map = dd.io.load(file)
