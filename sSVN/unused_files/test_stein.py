import numpy as np
import numpy.testing as npt
import h5py
import pytest
import os
import numdifftools as nd
from source.stein_experiments import GAUSSIAN, SVN, BILBY_TAYLORF2

# TODO h5 files can only be read when test script is run in tests directory. make it global
def testGaussianFlow():
    """
    Tests SVN on the original gaussian from papers github repo (with MAP removed!!!) This changes random seeds
    Returns:

    """
    with h5py.File('originalGaussFlow.h5', 'r') as hf:
        archivedParticles0 = hf['particles_0'][:]
        archivedParticles1 = hf['particles_1'][:]

    # Load model
    model = GAUSSIAN()
    model.DoF = 2
    model.nData = 1
    model.stdn = 0.3


    # Create SNV object that takes in posterior
    svn = SVN(model)

    # Set original SVN settings:
    svn.nParticles = 100
    svn.nIterations = 100
    svn.stepsize = 1

    # Run operations on SVN object
    svn.apply()

    currentParticles0 = svn.particles[0]
    currentParticles1 = svn.particles[1]

    npt.assert_allclose(currentParticles0, archivedParticles0)
    npt.assert_allclose(currentParticles1, archivedParticles1)

# %%
def testTaylorf2Likelihood():
    """
    Tests if a 2x2 grid used for plotting a likelihood contour behaves as expected
    """
    # Choose a point to inject
    inject = np.array([2.0, 2.0])

    # Create model
    bilby_model = BILBY_TAYLORF2(inject)

    # Choose grid length
    ngrid = 20

    # Create mesh (make sure to make entries floats, otherwise plots come out wrong.
    m1 = np.linspace(1.99, 2.01, ngrid)
    m2 = np.linspace(1.99, 2.01, ngrid)
    M1, M2 = np.meshgrid(m1, m2)

    # Calculate normalization
    scale_factor = bilby_model.logLikelihoodEvaluation(np.array([2, 2]))

    currentLikelihood = bilby_model.getMinusLogLikelihood(np.vstack((np.ndarray.flatten(M1), np.ndarray.flatten(M2)))).reshape(ngrid, ngrid) / scale_factor

    with h5py.File('taylorf2LikelihoodEvaluation.h5', 'r') as hf:
        archivedLikelihood = hf['20x20_taylorf2_likelihood_evaluation'][:]

    np.allclose(archivedLikelihood,currentLikelihood)


