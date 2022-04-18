# Finished methods go here

from __future__ import division, print_function

import sys
import autograd.numpy as np
import bilby
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure
from bilby import utils

sys.path.append('../')
outdir = 'outdir'
label = 'bns_example'
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Creates a global variable bilby_likelihood
# Inputs:
# inject - Array of injection parameters to create TaylorF2 waveform
# Outputs:
# Bilby likelihood object
def create_bilby_likelihood(inject):
    # --------------------------------------------------
    # Set up a random seed for result reproducibility.
    # --------------------------------------------------

    np.random.seed(88170235)

    # --------------------------------------------------
    # Define vector of injection parameters
    # --------------------------------------------------

    global injection_parameter_order

    injection_parameters = dict(
        mass_1 = inject[0],
        mass_2 = inject[1],
        chi_1 = 0.02,
        chi_2 = 0.02,
        luminosity_distance = 50.,
        theta_jn = 0.4,
        psi = 2.659,
        phase = 1.3,
        geocent_time = 1126259642.413,
        ra = 1.375,
        dec = -1.2108,
        lambda_1 = 400,
        lambda_2 = 450,
        iota = .1)

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

    # --------------------------------------------------
    # Set duration and sampling frequency for data segment injection.
    # TaylorF2 waveform cuts the signal close to the isco frequency
    # --------------------------------------------------

    duration = 8
    sampling_frequency = 2 * 1570.
    start_time = injection_parameters['geocent_time'] + 2 - duration

    waveform_arguments = dict(waveform_approximant='TaylorF2',
                              reference_frequency=50., minimum_frequency=40.0)

    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
        waveform_arguments=waveform_arguments)

    # --------------------------------------------------
    # Interferometer Setup
    # --------------------------------------------------

    # Default at design sensitivity and @ 40 Hz.
    # interferometers = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
    interferometers = bilby.gw.detector.InterferometerList(['H1'])
    for interferometer in interferometers:
        interferometer.minimum_frequency = 40

    #Data in interferometer is currently just noise governed by the PSD
    interferometers.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency, duration=duration,
        start_time=start_time)

    interferometers.inject_signal(parameters=injection_parameters,
                                  waveform_generator=waveform_generator)

    global bilby_likelihood

    bilby_likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=interferometers, waveform_generator=waveform_generator,
        time_marginalization=False, phase_marginalization=False,
        distance_marginalization=False)

# Converts bilby likelihood object to a function
# Inputs:
# theta_0 - Array of values to evaluate function at
# Outputs:
# Scalar value of log likelihood
def Bilby_BNS_Log_Likelihood_Function(theta_0):

    #only over m1 and m2 for now!
    theta = dict(
        mass_1=theta_0[0],
        mass_2=theta_0[1],
        chi_1=0.02,
        chi_2=0.02,
        luminosity_distance=50.,
        theta_jn=0.4, psi=2.659,
        phase=1.3,
        geocent_time=1126259642.413,
        ra=1.375,
        dec=-1.2108,
        lambda_1=400,
        lambda_2=450,
        iota=.1)

    bilby_likelihood.parameters=theta

    #print(theta_0)
    #print(bilby_likelihood.log_likelihood())

    return bilby_likelihood.log_likelihood()

# Functional form for the gradient
# Inputs:
# theta_0 - point at which to evaluate the gradient
# Outputs:
# theta_0 dimensional list representing the gradient

def grad_likelihood_BNS(theta_0):
    return utils.derivatives(theta_0,Bilby_BNS_Log_Likelihood_Function)


# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------



