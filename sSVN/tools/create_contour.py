from models.gauss_analytic import gauss_analytic
import numpy as np
import h5py
import os

import logging
# Instantiate logging
log = logging.getLogger(__name__)

def create_contour(stein, output_dir, begin=2, end=2):
    # Create a background contour file if necessary for given settings
    # begin = stein.model.begin
    # end = stein.model.end
    contour_file = "%s_%.2f_%.2f.h5" % (stein.model.id, begin, end)
    contour_file_path = '%s/%s' % (output_dir, contour_file)
    if not os.path.exists(contour_file_path):
        log.info('Contour file does not exist for given settings! Creating...')
        # ngrid = 1000
        # ngrid = 2000
        ngrid = 500
        x = np.linspace(-5, 4, ngrid)
        y = np.linspace(-5, 10, ngrid)
        # x = np.linspace(begin, end, ngrid)
        # y = np.linspace(begin, end, ngrid)
        X, Y = np.meshgrid(x, y)
        Z = np.exp(-1 * stein.model.getMinusLogPosterior_ensemble(np.vstack((np.ndarray.flatten(X), np.ndarray.flatten(Y))).T).reshape(ngrid,ngrid))
        # Z = np.exp( -1 * (stein.getMinusLogPosterior_ensemble(np.vstack((np.ndarray.flatten(X), np.ndarray.flatten(Y)))).reshape(ngrid, ngrid))) # Regular posterior plots
        log.info('Successfully created contour file for given settings.')
        with h5py.File(contour_file_path, 'a') as hf:
            hf.create_dataset("X", data=X)
            hf.create_dataset("Y", data=Y)
            hf.create_dataset("Z", data=Z)
    else:
        log.info('Using contour file for given settings found in /outdir ')
    return contour_file_path
