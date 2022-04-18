#%%
import bilby
import numpy as np
# import lalsimulation as lalsim

#%%
waveform_arguments = dict(waveform_approximant='TaylorF2', reference_frequency=50., minimum_frequency=40.0)

duration = 8
sampling_frequency = 2 * 1570.

parameters = dict(
    mass_1=36., mass_2=29., a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
    phi_12=1.7, phi_jl=0.3, luminosity_distance=2000., theta_jn=0.4, psi=2.659,
    phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)

#%%
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    waveform_arguments=waveform_arguments, parameters=parameters)

#%%

h = waveform_generator.frequency_domain_strain(parameters)
h_plus = h['plus']
h_cross = h['cross']
#%%
interferometers = bilby.gw.detector.InterferometerList(['L1'])
for interferometer in interferometers:
    interferometer.minimum_frequency = 40
#%%
# Data in interferometer is currently just noise governed by the PSD
interferometers.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration)
#%%
interferometers.inject_signal(parameters=parameters,
                              waveform_generator=waveform_generator)
#%%
bilby_likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=interferometers, waveform_generator=waveform_generator,
    time_marginalization=False, phase_marginalization=False,
    distance_marginalization=False)


#%%
from bilby.gw.source import lal_binary_black_hole

# lalsim.SimInspiralTaylorF2()
# lalsim.SimInspiralTaylorF2E

#%%
dict = {'a' : 50}