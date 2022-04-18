import logging
import numpy as np
from source.stein_experiments import SVI
from models.taylorf2_bilby import BILBY_TAYLORF2
import os
import h5py

# Instantiate logging
log = logging.getLogger(__name__)

def taylorf2_steindriver(iterations, particles, injection, particle_initialization, gradient_step = None,
                         hessian_step = None, rescale_likelihood = None, finite_difference_method = None, store_history = None,
                         rescale_prior = None, injection_sigma = None, stochastic_perturbation = 0, optimize_method = None):
    """
    Runs Stein on taylorF2 and produces h5 files with all information needed to create plots.
    Args:
        iterations: number of iterations (int)
        particles: number of particles (int)
        injection: list of injection parameters to simulate data (array)
        injection_sigma: list of standard deviations for parameter space
        particle_initialization: range in which to sample particles from gaussian prior (array)
        gradient_step: finite difference step size for gradient
        hessian_step: finite difference step size for hessian
        rescale_likelihood: scalar that divides the likelihood and rescales it (float)
        finite_difference_method: 'central' (str)
        store_history: store all intermediate distributions between beginning and final iterations (bool)
        rescale_prior: scalar that divides the likelihood and rescales it (float)

    Returns: Dictionary of absolute paths to output file: (1) h5_taylor_history, (2) h5_taylor_path

    """

    output_directory = os.path.dirname(os.path.abspath(__file__))  + '/outdir'

    # Create likelihood
    inject = np.array([injection[0], injection[1]])
    inject_sigma = np.array([injection_sigma[0], injection_sigma[1]])
    bilby_model = BILBY_TAYLORF2(inject)
    bilby_model.injectSigma = inject_sigma

    # Store argparse in local variables
    gstep = .1 if gradient_step is None else gradient_step
    hstep = .1 if hessian_step is None else hessian_step
    rescale_likelihood = bilby_model.logLikelihoodPeak if rescale_likelihood is None else rescale_likelihood # Default set to one in code.
    method = 'central' if finite_difference_method is None else finite_difference_method
    rescale_prior = 1 if rescale_prior is None else rescale_prior #bilby_model.logPriorPeak


    # Modify likelihood attributes to given settings
    bilby_model.hessianStep = np.float64(hstep)
    bilby_model.hessianMethod = method
    bilby_model.gradientStep = np.float64(gstep)
    bilby_model.gradientMethod = method
    bilby_model.scale_likelihood = np.float64(rescale_likelihood)
    bilby_model.scale_prior = rescale_prior

    # Run stein
    begin = particle_initialization[0]
    end = particle_initialization[1]
    particleRange = np.array([begin, end])
    optimizeMethod = optimize_method
    svn = SVI(bilby_model, optimizeMethod = optimizeMethod, particleRange = particleRange, nParticles = particles, nIterations = iterations,
              output_dir = output_directory)
    svn.stepsize = 1
    svn.store_history = False if store_history is None else store_history
    svn.stochastic_perturbation = stochastic_perturbation

    # Create a background contour file if necessary for given settings
    contour_file = "%s_rsl_%s_rsp_%s_%.2f_%.2f.h5" % (bilby_model.modelType, bilby_model.scale_likelihood, bilby_model.scale_prior, begin, end)
    contour_file_path = '%s/%s' % (output_directory, contour_file)
    if not os.path.exists(contour_file_path):
        log.info('Contour file does not exist for given settings! Creating...')
        ngrid = 100
        # ngrid = 500
        x = np.linspace(begin, end, ngrid)
        y = np.linspace(begin, end, ngrid)
        X, Y = np.meshgrid(x,y)
        Z = SVI.getMinusLogPosterior_ensemble(np.vstack((np.ndarray.flatten(X), np.ndarray.flatten(Y)))).reshape(ngrid, ngrid)
        # Z = np.exp( -1 * (bilby_model.getMinusLogPosterior_ensemble(np.vstack((np.ndarray.flatten(X), np.ndarray.flatten(Y)))).reshape(ngrid, ngrid))) # Regular posterior plots
        log.info('Successfully created contour file for given settings.')
        with h5py.File(contour_file_path, 'a') as hf:
            hf.create_dataset("X", data = X)
            hf.create_dataset("Y", data = Y)
            hf.create_dataset("Z", data = Z)
    else:
        log.info('Using contour file for given settings found in /outdir ')

    result_paths = {"h5_taylor_history" : svn.history_path, "h5_taylor_contours" : contour_file_path}

    try:
        svn.apply() # History file is produced by this method
    except Exception:
        log.error("Error occurred in SVN apply() method.", exc_info = True)
        log.error("[FAILED] - File with the following path was unsuccessful:\n %s" % result_paths["h5_taylor_history"])
        return result_paths
    else:
        log.info('[SUCCESS] - File with following path had a successful run:\n %s' % result_paths["h5_taylor_history"])
        return result_paths



