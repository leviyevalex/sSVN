import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import h5py
from time import time
import os

# Instantiate logging
log = logging.getLogger(__name__)

def animate_driver(contour_file_path, sampler):
    """
    Produces a particle flow animation. Takes in contour_file_path, which provides the background contourplot, and uses
    the particular sampler instance to obtain directories where results are stored.
    Args:
        contour_file_path: absolute file path to the likelihood contours (str)
        sampler: an instance of sampler after sampler.apply() executed  (sampler)

    Returns: Path to gif file.

    """

    # Read in data needed for static contourplot
    with h5py.File(contour_file_path, 'r') as hf:
        X = hf["X"][:]
        Y = hf["Y"][:]
        Z = hf["Z"][:]

    # Read in metadata
    with h5py.File(sampler.history_path, 'r') as hf:
        iterationsPerformed = hf['metadata']["L"][()]
        method = hf['metadata']['method'][()]
    frames = iterationsPerformed
    log.info('Animating %i frames' % frames)

    # Create figures folder
    animation_name = 'particle_flow.gif'
    animation_output_directory = os.path.join(sampler.RUN_OUTPUT_DIR, 'figures')
    if os.path.isdir(animation_output_directory) == False:
        os.mkdir(animation_output_directory)

    # Setup static figure
    fig, ax = plt.subplots(figsize = (5, 5))
    plt.axis('off')
    cp = ax.contour(X, Y, Z, 7, colors='black', alpha=0.1)
    scat = ax.scatter([], [], marker=".", color='#51AEFF', s=8) # Initial frame (empty scatterplot)

    with h5py.File(sampler.history_path, 'r') as hf:
        def update(i):
            index = np.arange(0, iterationsPerformed, 1) # Use to animate whole flow
            j = index[i]
            if i < iterationsPerformed:
                particles_x_i = hf['%i' % j]['X'][:, 0]
                particles_y_i = hf['%i' % j]['X'][:, 1]
            else:
                particles_x_i = hf['final_updated_particles']['X'][:, 0]
                particles_y_i = hf['final_updated_particles']['X'][:, 1]

            scat.set_offsets(np.c_[particles_x_i, particles_y_i])
            ax.set_title('%s particle flow: Frame %i' % (method, j))

        anim = FuncAnimation(fig, update, interval=3000 / iterationsPerformed, frames=frames,
                             repeat_delay=2000)

        animation_path = os.path.join(animation_output_directory, animation_name)
        anim.save(animation_path, writer='imagemagick')
        return animation_path