import os
import matplotlib
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
import numpy as np
from source.stein_experiments import SVI
from models.rosenbrock_analytic_proper_sample import rosenbrock_analytic
import deepdish as dd
import os
import pandas as pd
from celluloid import Camera
from matplotlib.animation import FuncAnimation
import matplotlib.image as mgimg
from send2trash import send2trash
from PIL import Image
import h5py
import logging
log = logging.getLogger(__name__)
import time
from scipy.sparse.linalg import *
#%%
CWD = os.getcwd()
# rel_file_path='/drivers/outdir/rosenbrock_proper_-2.50_2.50.h5'
rel_file_path='/drivers/first_system.h5'
# rel_file_path='/drivers/outdir/1596598591/rosenbrock_proper_nP_5000_1000_1596598591.h5'
abs_file_path = CWD + rel_file_path
# composite_map = dd.io.load(abs_file_path)

#%%
class test_solvers:
    def __init__(self):
        self.cgiter = 0
        self.dict_cg = {}
        dict = dd.io.load(abs_file_path)
        self.H_bar = dict['H_bar']
        self.mgJ = dict['mgJ']
        self.atol = None
    def callback(self,xk):
        # for cg iter debug
        x = np.copy(xk)
        self.cgiter += 1
        self.dict_cg[self.cgiter] = x

    def callback_reset(self):
        # for cg iter debug
        self.cgiter=0
        self.dict_cg={}
    def test_solver(self):
        self.callback_reset()
        start = time.time()
        res = lgmres(self.H_bar, self.mgJ, atol=self.atol, maxiter=4000, callback=self.callback)[0].reshape(3000, 2)
        end = time.time()
        total_time = end - start
        print('Average time per iteration: %f' % (total_time / self.cgiter))
        print('Solver took this long: %f' % (total_time))
        print('iterations: %i' % self.cgiter)
ts = test_solvers()
#%%
# ts.atol = None
# ts.atol = 1e-1
# ts.atol = 0.01
# ts.atol = 1e-2
ts.atol = 1e-3
# ts.atol = 1e-1
ts.test_solver()




#%%
# with h5py.File(abs_file_path, 'r') as hf:
#     X = hf["X"][:]
#     Y = hf["Y"][:]
#     Z = hf["Z"][:]
# plt.style.use('seaborn-white')
# # Setup static figure
# fig, ax = plt.subplots(figsize = (10, 10))
# ax.set_xlabel('mass_1')
# ax.set_ylabel('mass_2')
# ax.set_title('SVN particle flow', fontdict={'fontsize': 15})
# #%%
# cp = ax.contourf(X, Y, Z, 10, cmap="viridis")
# fig.colorbar(cp)
# plt.show()

