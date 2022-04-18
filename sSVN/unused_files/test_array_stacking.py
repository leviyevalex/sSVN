import numpy as np
import pytest
import numdifftools as nd
from source.stein_experiments import BILBY_TAYLORF2
################################################################
@pytest.fixture(scope = "module")
def bilby_for_stack_test():
    """Common class for stacking test"""
    inject = np.float64(np.array([10.0, 10.0]))
    model = BILBY_TAYLORF2(inject)
    model.hessianMethod = 'central'
    model.hessianStep = .01
    return model

def testStackedGradients(bilby_for_stack_test):
    """
    Tests if main gradient methods output arrays of the expected shape.
    """
    testParticles1 = np.array([[10., 9., 8.],
                               [10., 9., 8.]])

    col0 = np.array([10., 10.])
    col1= np.array([9., 9.])
    col2= np.array([8., 8.])

    dof = bilby_for_stack_test.DoF # Degrees of freedom, IE: number of parameters
    N = 3 # Number of particles in this case. Column (i) represents vector of coordinates for particle (i)

    dL = nd.Gradient(bilby_for_stack_test.logLikelihoodEvaluation, method = bilby_for_stack_test.gradientMethod, step = bilby_for_stack_test.gradientStep)
    dPrior = nd.Gradient(bilby_for_stack_test.getLogPrior, method = bilby_for_stack_test.gradientMethod, step = bilby_for_stack_test.gradientStep)

    prior_gradient_array = -1 * bilby_for_stack_test.getGradientMinusLogPrior(testParticles1)
    likelihood_gradient_array = -1 * bilby_for_stack_test.getGradientMinusLogLikelihood(testParticles1)

    # Expected shape
    assert(prior_gradient_array.shape == (dof, N))
    assert(likelihood_gradient_array.shape == (dof, N))

    # Expected data stacking in array
    assert(np.allclose(dPrior(col0), prior_gradient_array[:, 0]))
    assert(np.allclose(dPrior(col1), prior_gradient_array[:, 1]))
    assert(np.allclose(dPrior(col2), prior_gradient_array[:, 2]))

    assert(np.allclose(dL(col0), likelihood_gradient_array[:, 0]))
    assert(np.allclose(dL(col1), likelihood_gradient_array[:, 1]))
    assert(np.allclose(dL(col2), likelihood_gradient_array[:, 2]))

def testStackedHessians(bilby_for_stack_test):
    """
    Tests if hessian methods output arrays are of the expected shape.
    """
    testParticles1 = np.array([[10., 9., 8.],
                               [10., 9., 8.]])

    col0 = np.array([10., 10.])
    col1= np.array([9., 9.])
    col2= np.array([8., 8.])

    dof = bilby_for_stack_test.DoF # Degrees of freedom, IE: number of parameters
    N = 3 # Number of particles in this case. Column (i) represents vector of coordinates for particle (i)

    hessL = nd.Hessian(bilby_for_stack_test.logLikelihoodEvaluation, method = 'central', step = .01)
    hessPrior = nd.Hessian(bilby_for_stack_test.getLogPrior, method = 'central', step = .01)

    hessian_likelihood_array = -1 * bilby_for_stack_test.getGNHessianMinusLogLikelihood(testParticles1)
    hessian_prior_array = -1 * bilby_for_stack_test.getGNHessianMinusLogPrior(testParticles1)

    assert(hessian_prior_array.shape == (dof, dof, N))
    assert(hessian_likelihood_array.shape == (dof, dof, N))

    assert(np.allclose(hessPrior(col0), hessian_prior_array[:, :, 0]))
    assert(np.allclose(hessPrior(col1), hessian_prior_array[:, :, 1]))
    assert(np.allclose(hessPrior(col2), hessian_prior_array[:, :, 2]))

    assert(np.allclose(hessL(col0), hessian_likelihood_array[:, :, 0]))
    assert(np.allclose(hessL(col1), hessian_likelihood_array[:, :, 1]))
    assert(np.allclose(hessL(col2), hessian_likelihood_array[:, :, 2]))

def testStackedMethods(bilby_for_stack_test):
    """
    Tests if main method outputs produce arrays of the expected shape.
    """
    testParticles1 = np.array([[10., 9., 8.],
                               [10., 9., 8.]])

    col0 = np.array([10., 10.])
    col1= np.array([9., 9.])
    col2= np.array([8., 8.])

    dof = bilby_for_stack_test.DoF # Degrees of freedom, IE: number of parameters
    N = 3 # Number of particles in this case. Column (i) represents vector of coordinates for particle (i)

    # Testing data for log Prior

    prior_array = -1 * bilby_for_stack_test.getMinusLogPrior(testParticles1)
    likelihood_array = -1 * bilby_for_stack_test.getMinusLogLikelihood(testParticles1)

    assert(prior_array.shape) == (N,) # Expected shape
    assert(bilby_for_stack_test.getLogPrior(col0) == prior_array[0])
    assert(bilby_for_stack_test.getLogPrior(col1) == prior_array[1])
    assert(bilby_for_stack_test.getLogPrior(col2) == prior_array[2])

    assert(likelihood_array.shape) == (N,) # Expected shape
    assert(bilby_for_stack_test.logLikelihoodEvaluation(col0) == likelihood_array[0])
    assert(bilby_for_stack_test.logLikelihoodEvaluation(col1) == likelihood_array[1])
    assert(bilby_for_stack_test.logLikelihoodEvaluation(col2) == likelihood_array[2])
    ###
