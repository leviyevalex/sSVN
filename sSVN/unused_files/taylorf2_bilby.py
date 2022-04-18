import numpy as np
import copy
import bilby
import numdifftools as nd

class BILBY_TAYLORF2():
    def __init__(self, *arg):
        """
        Creates an object to input into SVN class.
        Args:
            *arg ():
            arg1 inject: [m1 > m2] necessary by convention (arg1 is optional)
            arg2 injectSigma: Gaussian std over inject variables for prior (optional)
        """
        self.modelType = 'taylorf2'
        self.randomSeed = 88170235  # Original from bilby tutorial. Seed used to create a unique likelihood
        self.inject = arg[0] if len(arg) > 0 else np.array([2, 2])
        self.injectSigma = arg[1] if len(arg) > 1 else np.array([.2, .2])
        self.injection_parameters = dict(
            mass_1=self.inject[0],
            mass_2=self.inject[1],
            chi_1=0.02,
            chi_2=0.02,
            luminosity_distance=120.,
            theta_jn=0.4,
            psi=2.659,
            phase=1.3,
            geocent_time=1126259642.413,
            ra=1.375,
            dec=-1.2108,
            lambda_1=400,
            lambda_2=450,
            iota=.1)
        self.params_copy = copy.deepcopy(self.injection_parameters) # used in the getWaveform method
        self.theta_0 = copy.deepcopy(self.injection_parameters)  # Will be used in likelihood evaluation
        self.DoF = self.inject.size
        self.injection_parameter_order = self.setInjectionParameterOrder()
        self.likelihood = self.create_bilby_likelihood()
        self.m1GaussianPrior = self.setGaussianPrior(self.inject[0], self.injectSigma[0])
        self.m2GaussianPrior = self.setGaussianPrior(self.inject[1], self.injectSigma[1])
        self.gradientStep = 1e-5  # Step size for central differencing the gradient
        self.gradientMethod = 'central'
        self.hessianStep = 1e-5  # Step size for central differencing the hessian
        self.hessianMethod = 'central'
        self.outOfBounds = 0  # What to do if particles pushed out of bounds. 0 = Return zero, 1 = Raise(ValueError), np.inf makes cost function infinite outside valid range
        self.scale_likelihood = 1  # Scalar used to rescale geometry of likelihood
        self.scale_prior = 1  # Scalar used to rescale geometry of prior
        self.logLikelihoodPeak = self.getMinusLogLikelihood_individual(self.inject)
        self.logPriorPeak = self.getMinusLogPrior_individual(self.inject)
        self.include_boundary = False # Includes term in posterior evaluation that will create a boundary around bad inputs
        self.forwardJacobian = nd.Jacobian(self.getWaveform, step=self.gradientStep, method=self.gradientMethod)

    def getWaveform(self, theta):
        """
        Get the gravitational waveform model given parameters

        theta: numpy array of parameters (m1, m2)
        """
        np.random.seed(self.randomSeed)

        # --------------------------------------------------
        # Set duration and sampling frequency for data segment injection.
        # TaylorF2 waveform cuts the signal close to the isco frequency
        # --------------------------------------------------

        duration = 8
        sampling_frequency = 2 * 1570.
        start_time = self.injection_parameters['geocent_time'] + 2 - duration

        waveform_arguments = dict(waveform_approximant='TaylorF2',
                                  reference_frequency=50., minimum_frequency=40.0)

        waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration, sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
            waveform_arguments=waveform_arguments, start_time = start_time)

        self.params_copy['mass_1'] = theta[0]
        self.params_copy['mass_2'] = theta[1]

        waveform_array = waveform_generator.frequency_domain_strain(parameters = self.params_copy)

        return np.real(waveform_array['plus'])

    def setInjectionParameterOrder(self):
        injection_parameter_order = [
            'mass_1',
            'mass_2',
            'chi_1',
            'chi_2',
            'luminosity_distance',
            'theta_jn',
            'psi',
            'phase',
            'geocent_time',
            'ra',
            'dec',
            'lambda_1',
            'lambda_2',
            'iota'
        ]
        return injection_parameter_order

    def create_bilby_likelihood(self):
        """
        Initializes a TAYLORF2 model using the Bilby inference library
        Returns: a Bilby likelihood object

        """
        np.random.seed(self.randomSeed)

        # --------------------------------------------------
        # Set duration and sampling frequency for data segment injection.
        # TaylorF2 waveform cuts the signal close to the isco frequency
        # --------------------------------------------------

        duration = 8
        sampling_frequency = 2 * 1570.
        start_time = self.injection_parameters['geocent_time'] + 2 - duration

        waveform_arguments = dict(waveform_approximant='TaylorF2',
                                  reference_frequency=50., minimum_frequency=40.0)

        waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration, sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
            waveform_arguments=waveform_arguments, start_time = start_time)

        # --------------------------------------------------
        # Interferometer Setup
        # --------------------------------------------------

        # Default at design sensitivity and @ 40 Hz.
        # interferometers = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
        interferometers = bilby.gw.detector.InterferometerList(['L1'])
        for interferometer in interferometers:
            interferometer.minimum_frequency = 40

        # Data in interferometer is currently just noise governed by the PSD
        interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency, duration=duration,
            start_time=start_time)

        interferometers.inject_signal(parameters=self.injection_parameters,
                                      waveform_generator=waveform_generator)

        bilby_likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers=interferometers, waveform_generator=waveform_generator,
            time_marginalization=False, phase_marginalization=False,
            distance_marginalization=False)

        return bilby_likelihood

    # 1
    def getMinusLogLikelihood_individual(self, theta):
        """
        Evaluates the likelihood at a particular point in parameter space
        The previous dictionary copy is just to make code pretty.
        Args:
            theta (d dimensional np.array): Where to evaluate the likelihood in parameter space
        Returns: scalar

        """
        # # only over m1 and m2 for now!
        #
        if theta[0] <= 0 or theta[1] <= 0:
            if self.outOfBounds == 1:
                message = f"\n Input to logLikelihoodEvaluation outside of domain \n mass_1 = {theta[0]} or mass_2 = {theta[1]} <= 0 "
                raise ValueError(message)
            elif self.outOfBounds == 0:
                return 0
            elif self.outOfBounds == np.inf:
                return np.inf

        for pos, var in enumerate(self.injection_parameter_order):
            if pos < self.DoF:
                self.theta_0[var] = theta[pos]
            else:
                break

        self.likelihood.parameters = self.theta_0

        return -1 * self.likelihood.log_likelihood() / self.scale_likelihood

    def setGaussianPrior(self, theta_i, sigma_i):
        """
        Helper function that sets a gaussian prior using the bilby library
        Args:
            theta_i: mean of the gaussian distribution (float)
            sigma_i: standard deviation (float)

        Returns: bilby gaussian

        """

        return bilby.core.prior.Gaussian(theta_i, sigma_i)

    # 2
    def getMinusLogPrior_individual(self, theta):
        """
        Helper function to get priors. Assume parameters are independent and distributed via gaussian
        Args:
            theta (): d dimensional array representing position in parameter space

        Returns: scalar.

        """

        if theta[0] <= 0 or theta[1] <= 0:
            if self.outOfBounds == 1:
                message = f"\n Input to getLogPrior outside of domain \n mass_1 = {theta[0]} or mass_2 = {theta[1]} <= 0 "
                raise ValueError(message)
            elif self.outOfBounds == 0:
                pass
                return 0
            elif self.outOfBounds == np.inf:
                return np.inf

        return -1 * (self.m1GaussianPrior.ln_prob(theta[0]) + self.m2GaussianPrior.ln_prob(theta[1]))/self.scale_prior

    # 3
    def getGradientMinusLogPrior_individual(self, theta):
        """
        Helper function to get gradient of minus log prior for individual particle
        Args:
            theta: particle location (d x 1 dimensional array)

        Returns: gradient evaluated at particle position (d dimensional array)

        """
        return nd.Gradient(self.getMinusLogPrior_individual, step=self.gradientStep, method=self.gradientMethod)(theta)

    # 4
    def getGradientMinusLogLikelihood_individual(self, theta):
        """
        Helper function to evaluate gradient of minus log likelihood at a particle
        Args:
            theta: particle position (d x 1 dimensional array)

        Returns: gradient of minus log likelihood evaluated at a particle (d dimensional array)

        """
        return nd.Gradient(self.getMinusLogLikelihood_individual, step=self.gradientStep, method=self.gradientMethod)(theta)

    # 5
    def getHessianMinusLogPrior_individual(self,theta):
        """
          Method to get hessian of negative log prior
          Args:
              theta: particle position (d x 1 array)

          Returns: hessian evaluated at each particle position (d x d x 1 array)

          """

        return nd.Hessian(self.getMinusLogPrior_individual, method=self.hessianMethod, step=self.hessianStep)(theta)

    # 6
    def getHessianMinusLogLikelihood_individual(self,theta):
        """
        Helper method to get hessian of minus log likelihood for individual particle
        Args:
            theta: particle position (n x 1 dimensional array)

        Returns: hessian of minus log likelihood evaluated at particle position

        """
        return nd.Hessian(self.getMinusLogLikelihood_individual, method=self.hessianMethod, step=self.hessianStep)(theta)

