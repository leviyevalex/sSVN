#%%
import numpy as np
import os
# import dask.array as da
import h5py
import matplotlib.pyplot as plt
# import deepdish as dd
import os
import logging.config
from pathlib import Path
from plots.plot_helper_functions import set_size
import scipy
from matplotlib import cm
#%%
log = logging.getLogger(__name__)
def create_particle_trajectory_plots(output_dir=None, contour_file_path=None):
    log.info('Making particle trajectory plots:')
    if output_dir is None and contour_file_path is None:
        root = os.path.dirname(os.path.abspath(__file__))
        read_file = os.path.join(root, 'double-banana-sSVGD-500.h5') # read from
        # read_file = os.path.join(root, 'double-banana-sSVN-500.h5') # read from
        save_file = os.path.join(root, 'particle_trajectories.png') # store to
        contour_file_path = os.path.join(Path(os.getcwd()).parent, 'outdir', 'rosenbrock_proper_-2.50_2.50.h5')
    else:
        read_file = os.path.join(output_dir, 'output_data_new.h5')
        figures_directory = os.path.join(output_dir, 'figures')
        save_file = os.path.join(figures_directory, 'particle_trajectories.png')
        if os.path.isdir(figures_directory) == False:
            os.mkdir(figures_directory)
    with h5py.File(contour_file_path, 'r') as hf:
        X = hf["X"][:]
        Y = hf["Y"][:]
        Z = hf["Z"][:]

    width = 469.75502
    fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
    # fig, ax = plt.subplots(figsize = (10, 10))
    ax.set_facecolor('#FFFFFF')
    fig.patch.set_facecolor('#FFFFFF')
    plt.axis('off')
    cp = ax.contour(X, Y, Z, 7, colors='black', alpha=0.1)
    with h5py.File(read_file, 'r') as hf:
        iters_performed = hf['metadata']['L'][()]
        nParticles = hf['metadata']['nParticles'][()]
        # range_truncated = np.arange(iters_performed - 500, iters_performed)
        for l in np.arange(iters_performed/2, iters_performed, 10):
        # for l in range_truncated:
            particles = hf['%i' % l]['X'][()]
            particles_x = particles[:, 0]
            particles_y = particles[:, 1]
            # alpha = 1 - (float(l) / nParticles)
            alpha =  (l / (iters_performed - iters_performed/2) /10) * 3.5/5
            # print(alpha)
            # assert (alpha >= 0 and alpha <= 1)
            ax.scatter(particles_x, particles_y, marker=".", color='#51AEFF', s=0.01, alpha=alpha)
        X_final = hf['final_updated_particles']['X'][()]
        ax.scatter(X_final[:, 0], X_final[:, 1], marker="x", s=0.01, color='#1291ff')
        # color='#51AEFF'
    fig.savefig(save_file)
    log.info('INFO: Successfully created moment convergence plot')

def main():
    create_particle_trajectory_plots()
if __name__ == '__main__':
    main()
