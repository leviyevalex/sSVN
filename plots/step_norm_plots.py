import numpy as np
import os
# import dask.array as da
# import h5py
import matplotlib.pyplot as plt
# import deepdish as dd
import os
def create_step_norm_plots(metadata, output_dir):
    animation_output_directory = output_dir + '/figures'
    if os.path.isdir(animation_output_directory) == False:
        os.mkdir(animation_output_directory)

    def extract_step_norm(metadata):
        last_iter = list(metadata.keys())[-2]
        step_norm = np.array([])
        for l in range(last_iter):
            step_norm_l = metadata[l]['step_norm']
            step_norm = np.append(step_norm, step_norm_l)
        return np.log10(step_norm)
    def extract_phi(metadata):
        last_iter = list(metadata.keys())[-2]
        phi = np.array([])
        for l in range(last_iter):
            phi_l = metadata[l]['phi']
            phi = np.append(phi, phi_l)
        return np.log10(phi)
    logphi = extract_phi(metadata)
    logstep_norm = extract_step_norm(metadata)
    plt.style.use('seaborn-dark-palette')
    plt.style.use('seaborn-dark')
    fig, ax = plt.subplots()
    ax.plot(logstep_norm, label='Log(step_norm)')
    ax.plot(logphi, label='Log(phi)')
    plt.legend()
    ax.set_title("Convergence Criteria")
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Log Value')
    filename = output_dir + 'figures/' + 'step_norm.png'
    fig.savefig(filename)