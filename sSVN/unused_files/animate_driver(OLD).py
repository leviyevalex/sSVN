import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import h5py
from time import time
import os
import argparse

# Instantiate logging
log = logging.getLogger(__name__)

def animate_driver(contour_file_path, particle_history_path, output_dir):
    """
    Produces a particle flow animation
    Args:
        contour_file_path: absolute file path to the likelihood contours (str)
        particle_history_path: absolute file path to the particle distribution history (str)

    Returns: Particle flow animation stored in /figures.

    """
    # Read in info to plot static contour background on animation

    # output_directory = os.path.dirname(os.path.abspath(__file__))  + '/figures'
    animation_output_directory = output_dir + '/figures'

    if os.path.isdir(animation_output_directory) == False:
        os.mkdir(animation_output_directory)

    now = str(int(time()))
    animation_name = '%s.gif' % now

    # Read in data needed for static contour
    with h5py.File(contour_file_path, 'r') as hf:
        X = hf["X"][:]
        Y = hf["Y"][:]
        Z = hf["Z"][:]

    # Access metadata and store in list of tuples from particle_history_path
    with h5py.File(particle_history_path, 'r') as hf:
        group = hf["particle_history"]
        attributes_list = list(group.attrs.items())
        # Used to feed into FuncAnimation
        history = []
        group.visit(history.append)
        iterationsPerformed = len(history)
        particles_x_0 = hf["particle_history/%s" % history[0]][0, :]
        particles_y_0 = hf["particle_history/%s" % history[0]][1, :]
    # Setup static figure
    fig, ax = plt.subplots(figsize = (10, 10))
    ax.set_xlabel('mass_1')
    ax.set_ylabel('mass_2')
    optimizeMethod = 'SVN'
    ax.set_title('%s particle flow' % optimizeMethod, fontdict={'fontsize': 20})

    ### Textbox settings ###############################################################
    textstr = ''
    for tuple in attributes_list:
        if tuple[0] == 'nIterations':
            nIterations = tuple[1]  # Pick up this value for number of frames in animation
        if tuple[0] != 'filename':
            if tuple == attributes_list[-1]:
                textstr += '%s: %s' % (tuple[0], tuple[1])
            else:
                textstr += '%s: %s \n' % (tuple[0], tuple[1])

    # matplotlib.patch.Patch properties
    props = dict(boxstyle='square', facecolor='green', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    for tuple in attributes_list:
        if tuple[0] == 'nIterations':
            nIterations = tuple[1] # Pick up this value for number of frames in animation
        if tuple[0] != 'filename':
            if tuple == attributes_list[-1]:
                textstr += '%s: %s' % (tuple[0], tuple[1])
            else:
                textstr += '%s: %s \n' % (tuple[0], tuple[1])

    # matplotlib.patch.Patch properties
    props = dict(boxstyle='square', facecolor='green', alpha=0.5)

    # place a text box in upper left in axes coords
    # ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
    #         verticalalignment='top', bbox=props)
    ####################################################################################

    cp = ax.contourf(X, Y, Z, 10)
    fig.colorbar(cp)
    scat = ax.scatter([], [], marker="x", color='r') # Initial frame (empty scatterplot)
    ax.scatter(particles_x_0, particles_y_0, marker='+', color='g')

    # Update frame
    def update(i):
        with h5py.File(particle_history_path, 'r') as hf:
            group = hf["particle_history"]
            history = []
            group.visit(history.append)
            particles_x_i = hf["particle_history/%s" % history[i]][0, :]
            particles_y_i = hf["particle_history/%s" % history[i]][1, :]
            scat.set_offsets(np.c_[particles_x_i, particles_y_i])

        ax.set_title('SVN particle flow Frame %i' % i, fontdict={'fontsize': 20})
    anim = FuncAnimation(fig, update, interval = 3000 / iterationsPerformed, frames = iterationsPerformed,
                         repeat_delay = 2000)

    # plt.draw()
    # plt.show()

    anim.save('%s/%s' % (animation_output_directory, animation_name), writer = 'imagemagick')

def main():
    parser = argparse.ArgumentParser(description='Create animation using a background contour and particle flow data.')
    parser.add_argument('-fcp', '--file_contour_path', type=str, required=False, help='Path to h5 contour file')
    parser.add_argument('-fhp', '--file_history_path', type=str, required=False, help='Path to h5 history file ')
    args = parser.parse_args()
    animate_driver(contour_file_path = args.file_contour_path, particle_history_path = args.file_history_path)

if __name__ == "__main__":
    main()
