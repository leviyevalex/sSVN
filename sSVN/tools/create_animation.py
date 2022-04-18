import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import h5py
from time import time
import os
import deepdish as dd
import argparse

# Instantiate logging
log = logging.getLogger(__name__)

def animate_driver(contour_file_path, RUN_OUTPUT_DIR):
    """
    Produces a particle flow animation
    Args:
        contour_file_path: absolute file path to the likelihood contours (str)
        particle_history_path: absolute file path to the particle distribution history (str)

    Returns: Particle flow animation stored in /figures.

    """
    # Read in info to plot static contour background on animation

    # output_directory = os.path.dirname(os.path.abspath(__file__))  + '/figures'
    # animation_output_directory = output_dir + '/figures'
    animation_output_directory = os.path.join(RUN_OUTPUT_DIR, 'figures')
    if os.path.isdir(animation_output_directory) == False:
        os.mkdir(animation_output_directory)

    # now = str(int(time()))

    # Read in data needed for static contour
    with h5py.File(contour_file_path, 'r') as hf:
        X = hf["X"][:]
        Y = hf["Y"][:]
        Z = hf["Z"][:]

    history_path = os.path.join(RUN_OUTPUT_DIR, 'output_data_new.h5')
    composite_map = dd.io.load(history_path)
    # method = composite_map['metadata']['optimization_method']
    iterationsPerformed = composite_map['metadata']['L']
    # Setup static figure
    fig, ax = plt.subplots(figsize = (10, 10))
    # ax.set_xlabel('mass_1')
    # ax.set_ylabel('mass_2')
    ax.set_title('particle flow', fontdict={'fontsize': 20})
    ax.set_facecolor('#F5FEFF')
    fig.patch.set_facecolor('#F5FEFF')
    plt.axis('off')
    cp = ax.contour(X, Y, Z, 7, colors='black', alpha=0.1)
    # cp = ax.contourf(X, Y, Z, 10, cmap='viridis')
    # fig.colorbar(cp)
    scat = ax.scatter([], [], marker=".", color='#51AEFF', s=8) # Initial frame (empty scatterplot)
    if composite_map != None:
        animation_name = 'basis_set.gif'

        # if iterationsPerformed < 200:
        #     frac = 5 / 10
        # elif iterationsPerformed < 500:
        #     frac = 3 / 10
        # elif iterationsPerformed < 1000:
        #     frac = 1 / 10
        # else:
        #     frac = .1/10
        # frames = int(np.floor((iterationsPerformed * (frac))))
        # if iterationsPerformed > 100:
        #     frames = 100
        # else:
        frames = iterationsPerformed
        print('Animating %i frames' % frames)
        # frames = (iterationsPerformed+1)
        def update(i):
            # log.info('updated %i' % i)
            index = np.arange(iterationsPerformed - frames, iterationsPerformed, 1) # Use to animate tail
            # index = np.arange(0, iterationsPerformed, 1) # Use to animate whole flow
            j = index[i]
            if i < iterationsPerformed:
                particles_x_i = composite_map['%i' % j]['X'][:, 0]
                particles_y_i = composite_map['%i' % j]['X'][:, 1]
            else:
                particles_x_i = composite_map['final_updated_particles']['X'][:, 0]
                particles_y_i = composite_map['final_updated_particles']['X'][:, 1]

            scat.set_offsets(np.c_[particles_x_i, particles_y_i])
            ax.set_title('particle flow Frame %i' % (j), fontdict={'fontsize': 20})

        # iterationsPerformed = sorted(composite_map.keys())[-1]

        anim = FuncAnimation(fig, update, interval=3000 / iterationsPerformed, frames=frames,
                             repeat_delay=2000)

        animation_file = os.path.join(animation_output_directory, animation_name)
        anim.save(animation_file, writer='imagemagick')

    # if test_set_dict != None:
    #     animation_name = '%i_test_set.gif' % m
    #     def update(i):
    #         particles_x_i = test_set_dict[i]['%i' % m][:, 0]
    #         particles_y_i = test_set_dict[i]['%i' % m][:, 1]
    #         scat.set_offsets(np.c_[particles_x_i, particles_y_i])
    #         ax.set_title('%s particle flow Frame %i' % (method, i), fontdict={'fontsize': 20})
    #
    #     iterationsPerformed = sorted(test_set_dict.keys())[-1]
    #     anim = FuncAnimation(fig, update, interval=3000 / iterationsPerformed, frames=iterationsPerformed,
    #                          repeat_delay=2000)
    #
    #     anim.save('%s/%s' % (animation_output_directory, animation_name), writer='imagemagick')

