#%%
import os
# import seaborn as sns; sns.set(color_codes=True)
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from source.stein_experiments import SVI
from models.rosenbrock_analytic_proper_sample import rosenbrock_analytic
import deepdish as dd
import logging
from matplotlib.animation import FuncAnimation
import copy
import seaborn as sns; sns.set(style="ticks", color_codes=True)
logger = logging.getLogger(__name__)
#%%
class displacement_field_plots:
    def __init__(self, stein, run_output_path, h5_file_path = None):
        self.save_path = run_output_path + 'figures'
        assert((h5_file_path != None and stein != None) == False)
        if h5_file_path != None:
            self.composite_transport_map = dd.io.load(self.h5_file_path)
        else:
            self.composite_transport_map = stein.composite_transport_map
        self.output_dir = os.path.dirname(os.path.abspath(__file__)) + '/outdir/'
        self.stein = stein
        self.h5_file_path = h5_file_path
        self.iter_data = {}
        self.last_iter = list(self.composite_transport_map.keys())[-1]
        self.filename = '/transport_field.gif'
        self.init_grid_particles()

    def init_grid_particles(self):
        self.begin = 2.5 * self.stein.model.begin
        self.end = 2.5 * self.stein.model.end
        self.ngrid = 100
        x = np.linspace(self.begin, self.end, self.ngrid)
        y = np.linspace(self.begin, self.end, self.ngrid)
        self.X, self.Y = np.meshgrid(x, y)
        self.grid_particles = np.vstack((np.ndarray.flatten(self.X), np.ndarray.flatten(self.Y))).T

    def init_iter_data_dict(self):
        iter_dict = {}
        iter_dict['displacement_field'] = None
        iter_dict['particles'] = None
        return iter_dict

    def get_pushforward_info(self, ensemble):
        thetas = np.copy(ensemble)
        vector_field = np.zeros((thetas.shape[0], thetas.shape[1]))
        for l in range(self.last_iter):
            eps = self.composite_transport_map[l]['eps']
            M = self.composite_transport_map[l]['metric']
            h = self.composite_transport_map[l]['bandwidth']
            alphas = self.composite_transport_map[l]['alphas']
            X = self.composite_transport_map[l]['X']
            set = {}
            set[0] = thetas
            if l == 0:
                self.stein.iter_ = 0
            elif l != 0:
                self.stein.iter_ = 1
            kernel = self.stein.getKernelWhole(bandwidth=h, metric=M, X=X, T_dict=set)[0]
            w = self.stein.w(alphas=alphas, kernel=kernel)
            thetas += eps * w
            vector_field += eps * w
            self.iter_data[l] = self.init_iter_data_dict()
            self.iter_data[l]['displacement_field'] = copy.deepcopy(vector_field)
            self.iter_data[l]['particles'] = copy.deepcopy(thetas)

    def update(self, i):
        vf = self.iter_data[i]['displacement_field']
        # pushed =
        # x = pushed[:, 0]
        # y = pushed[:, 1]
        # X, Y = np.meshgrid(x, y)
        U = vf[:, 0]
        V = vf[:, 1]
        self.Q.set_UVC(U,V)
        return self.Q,

    def animate_displacement_fields(self):
        fig, ax = plt.subplots(figsize=(9, 9))
        vf = self.iter_data[0]['displacement_field']
        U = vf[:, 0]
        V = vf[:, 1]
        self.Q = plt.quiver(self.X, self.Y, U, V, np.random.rand(self.ngrid ** 2), angles='xy', scale_units='xy', scale=1, alpha=0.4)
        plt.axis('off')
        anim = FuncAnimation(fig, self.update, blit=True, frames=self.last_iter)
        anim.save(self.save_path + self.filename, writer='imagemagick')

