from __future__ import division, print_function
import numpy as np
import bilby
import matplotlib.pyplot as plt
from bilby.gw.likelihood import get_binary_black_hole_likelihood as bbhl, BasicGravitationalWaveTransient
import numdifftools as nd
import scipy.stats as stats
from scipy.stats import truncnorm
from scipy.stats import truncnorm

#%%
# Attributes that end with 'X are not stored in metadata. Necessary for h5 storage to work!!!
# Note: bgwt, hanford, inf, etc... are not simple to store. end them with X
class imr_phenom:
    def __init__(self, *arg):

        self.setup()

        self.modelType = 'imr_phenom'

        self.DoF = 2

        # Range to plot
        self.begin = 30
        self.end = 40

        # Range to sample from
        self.low = 30
        self.high = 40

        # For numerical derivatives
        self.gradientStep = 1e-5  # Step size for central differencing the gradient
        self.gradientMethod = 'central'

        self.sigmaInv = np.linalg.inv(np.array([[0.5, 0],
                                                [0, 0.5]]))

        self.mu = np.array([35, 35])

    def setup(self):
        self.injection_parameters = dict(mass_1=35., mass_2=35., a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
                                    phi_12=1.7, phi_jl=0.3, luminosity_distance=2000., theta_jn=0.4, psi=2.659,
                                    phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)

        # Waveform generator for injection
        waveform_arguments = dict(waveform_approximant='IMRPhenomPv2', reference_frequency=50., minimum_frequency=20.)

        np.random.seed(1)

        duration = 4.
        sampling_frequency = 2048.
        self.waveform_generator_X = bilby.gw.WaveformGenerator(duration=duration, sampling_frequency=sampling_frequency,
                                                        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                                                        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                                                        waveform_arguments=waveform_arguments)

        # Create an interferometer object
        ifos = bilby.gw.detector.InterferometerList(['H1'])

        # Injected noise contribution to strain (actually depends on the PSD-sensitivity of the detector)
        start_time = self.injection_parameters['geocent_time'] - 3

        # ifos.set_strain_data_from_zero_noise(sampling_frequency=sampling_frequency, duration=duration, start_time=start_time)

        ifos.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency, duration=duration,start_time=start_time)

        # Injected model contribution to strain
        ifos.inject_signal(waveform_generator=self.waveform_generator_X, parameters=self.injection_parameters)

        self.injected_signal = self.hanford_X.frequency_domain_strain

        self.hanford_X = ifos[0]
        self.bgwt_X = BasicGravitationalWaveTransient(interferometers=ifos, waveform_generator=self.waveform_generator_X)

    def getInjectedSignal(self):
        return self.injected_signal  # Exactly as used in likelihood

    def forwardMinusInjection(self, theta):
        return self.getForward(theta) - self.injected_signal.real


    def getPSD(self):
        """
        Gets power spectral density
        Returns:

        """
        return self.hanford_X.power_spectral_density_array  # Exactly as used in likelihood

    def getForward(self, theta):
        """
        Forward solve
        Args:
            theta: particle positions d dim array

        Returns:

        """
        forward_parameters = dict(mass_1=theta[0], mass_2=theta[1], a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
                                     phi_12=1.7, phi_jl=0.3, luminosity_distance=2000., theta_jn=0.4, psi=2.659,
                                     phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)

        polarizations = self.waveform_generator_X.frequency_domain_strain(forward_parameters)

        # requires parameters because it needs extrinsic parameters
        return self.hanford_X.get_detector_response(waveform_polarizations=polarizations, parameters=forward_parameters).real
        # return self.hanford_X.inject_signal(waveform_generator=self.waveform_generator_X, parameters=forward_parameters)

    def gradForward(self, theta):
        return nd.Gradient(self.getForward, step=self.gradientStep, method=self.gradientMethod)(theta)

    # 1
    def getMinusLogLikelihood_individual(self, theta):
        assert(self.DoF == theta.size)

        likelihood_parameters = dict(mass_1=theta[0], mass_2=theta[1], a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
                                     phi_12=1.7, phi_jl=0.3, luminosity_distance=2000., theta_jn=0.4, psi=2.659,
                                     phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)
        self.bgwt_X.parameters = likelihood_parameters

        return -1 * self.bgwt_X.log_likelihood()

    # 2
    def getGradientMinusLogLikelihood_individual(self, theta):
        return nd.Gradient(self.getMinusLogLikelihood_individual, step=self.gradientStep, method=self.gradientMethod)(theta)

    # 3
    def getGNHessianMinusLogLikelihood_individual(self,theta):
        delta_T = self.waveform_generator_X.duration
        tiled_PSD = np.tile(self.getPSD(), (2, 1)).T
        grad_forward = self.gradForward(theta)
        return 4 * np.einsum('fd, fb -> db', np.conj(grad_forward), grad_forward / (tiled_PSD * delta_T))
        # return np.outer(grad_forward, grad_forward / self.getPSD())

######################################################################################
##### Flat priors
######################################################################################
    # # 4
    # def getMinusLogPrior_individual(self, theta):
    #     return 1
    # # 5
    # def getGradientMinusLogPrior_individual(self, theta):
    #     return 0
    # # 6
    # def getGNHessianMinusLogPrior_individual(self,theta):
    #     return 0
    # # 7
    # def newDrawFromPrior(self, nParticles):
    #     np.random.seed(1)
    #
    #     particles = np.random.uniform(low = self.low, high = self.high, size = self.DoF).reshape(self.DoF, 1)
    #     while particles.shape[1] != nParticles:
    #         particles = np.hstack((particles, (np.random.uniform(low=self.low, high=self.high, size=self.DoF)).reshape(self.DoF, 1)))
    #     return particles
#########################################################################################
    # 7


    def getMinusLogPrior_individual(self, theta):
        # term = 1/2 * (theta - self.mu).T @ self.SigmaInv @ (theta - self.mu)
        # assert(term.type)
        temp = (theta-self.mu)
        return 1/2 * (temp).T @ self.sigmaInv @ (temp)
    def getGradientMinusLogPrior_individual(self, theta):
        temp = (theta-self.mu)
        return 1/2 * self.sigmaInv @ (temp) + 1/2 * (temp).T @ self.sigmaInv
    def getGNHessianMinusLogPrior_individual(self, theta):
        return self.sigmaInv

    def newDrawFromPrior(self, nParticles):
        return np.random.multivariate_normal(mean=self.mu, cov=self.sigmaInv, size=nParticles).T

    # def newDrawFromPrior(self, nParticles):
    #     np.random.seed(1)
    #     return np.random.multivariate_normal(mean=self.mu, cov=self.sigmaInv, size=nParticles).T
    #
    # def getMeanCov(self):







#%%




# np.random.seed(1)
#%%






#%%
# Provides the polarizations I think
# h = waveform_generator.frequency_domain_strain()
# h_plus = h['plus']
# h_cross = h['cross']


# response = ifos[0].get_detector_response(h, injection_parameters)

# strain_data = ifos[0].strain_data
# strain_data_freq_array = strain_data.frequency_array
#
# # fig, ax = plt.subplots()
# a = bilby.gw.likelihood
# # fig = plt.figure()
# # plt.plot(psd)
# # fig.show()
# like = bbhl(ifos[0])
# # ax.plot(response)
# # ax.plot(h_plus)
# # plt.plot(h)
# # plt.plot(response)
# plt.plot(strain_data)
# #%%
# plt.show()
#
# # fig, ax = plt.subplots(figsize=(5, 3))
# # ax.plot(response)
# # fig.show()