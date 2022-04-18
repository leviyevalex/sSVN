from time import time
import h5py
import os
import logging
import numpy as np
log = logging.getLogger(__name__)
def initStorageFilenameH5(modelType, nParticles, nIterations):
    """
    Used to define the output file name.
    Returns: output file name (str)

    """
    now = str(int(time()))
    output_file_name = "%s_nP_%i_%i_%s.h5" \
                       % (modelType,
                          nParticles,
                          nIterations,
                          now)

    return output_file_name

def storeModelMetadataH5(model, output_file_path):
    """
    Stores metadata in output_file_path under group
    Args:
        model: get metadata from this class
        group: which group in h5 file to store data in
        output_file_path: path to h5 file

    Returns: Nothing

    """
    group = 'metadata'
    if os.path.isfile(output_file_path):
        log.warning('History h5 file with these settings already exists. Removing...')
        os.remove(output_file_path)
        log.warning('Successfully removed previous h5 history file.')
    with h5py.File(output_file_path, 'a') as hf:
        # All attributes/metadata to store into the h5 output file.
        hf.create_group(group)
        for attribute, value in model.__dict__.items():
            if attribute[-1] != 'X': # do not store attributes that end with 'X
                hf[group].attrs['%s' % attribute] = value

    log.info('Metadata for current run stored successfully!')

def storeIterationDataH5(iter_, nIterations, output_file_path, ensemble, mean, variance):
    """
    Stores particle distribution at each iteration into h5 file
    Args:
        iter_: current iteration. Needed to label dataset
        nIterations: Total number of iterations
        output_file_path: where to store file
        The rest are data.
    """

    # Store beginning and final distribution. If store_history = True, store all distributions
    num = int(len(str(nIterations)))
    group_history = "particle_history"
    mean_history = 'mean_history'
    variance_history = 'variance_history'
    with h5py.File(output_file_path, 'a') as hf:
        label = str(iter_).zfill(num)  # Padded label. 001, 002, 003, etc...
        hf.create_dataset("%s/%s_iteration" % (group_history, label), data=ensemble)
        hf.create_dataset("%s/%s_mean_norm" % (mean_history, label), data=mean)
        hf.create_dataset("%s/%s_var_norm" % (variance_history, label), data=variance)

def storeResults(dict, output_file_path):
    # group_result = 'dictionaries'
    with h5py.File(output_file_path, 'a') as hf:
        # label = str(iter_).zfill(num)  # Padded label. 001, 002, 003, etc...
        hf.create_dataset("Result", data=dict)



def getFinalEnsemble(particle_history_path):
    """
    Get ensemble locations from final iteration stored in h5 file
    Args:
        particle_history_path: h5 file from which to pull ensemble data

    Returns: ensemble positions

    """
    with h5py.File(particle_history_path, 'r') as hf:
        group = hf["particle_history"]
        # attributes_list = list(group.attrs.items())
        # Used to feed into FuncAnimation
        history = []
        group.visit(history.append)
        iterationsPerformed = len(history)
        return hf["particle_history/%s" % history[-1]][...]

def getMomentData(particle_history_path):
    with h5py.File(particle_history_path, 'r') as hf:
        group_mean = hf["mean_history"]
        group_var = hf["var_history"]
        # attributes_list = list(group.attrs.items())
        # Used to feed into FuncAnimation
        history_mean = []
        history_var = []
        group_mean.visit(history_mean.append)
        group_var.visit(history_var.append)
        iterationsPerformed = len(history_mean)
        mean = np.zeros(iterationsPerformed)
        var = np.zeros(iterationsPerformed)
        for iter in range(iterationsPerformed):
            mean_iter = hf["mean_history/%s" % history_mean[iter]][...]
            var_iter = hf["var_history/%s" % history_var[iter]][...]
            mean[iter] = mean_iter
            var[iter] = var_iter
        return mean, var
