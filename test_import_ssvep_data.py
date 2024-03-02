#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 22:02:31 2024

test_import_ssvep_data.py

This file serves as the test script for BCI Spring 2024 Lab 03. The functions defined in the module are called to evaluate the SSVEP data for subject 1 in both the time and frequency domains. Effectively, this script generates the plots that allow for the visualization of the data and saves them locally as image files. The production of this data allows for a deeper analysis of what the data truly represent, provided in a multiline comment at the end of this script.

Useful abbreviations:
    EEG: electroencephalography
    SSVEP: steady-state visual evoked potentials
    fs: sampling frequency
    FFT: Fast Fourier Transform

@author: Peijin Chen and Claire Leahy

"""

# import applicable functions from module
from import_ssvep_data import load_ssvep_data, plot_raw_data, epoch_ssvep_data, get_frequency_spectrum, plot_power_spectrum

#%% Part 1: Load the Data

# load data for subject 1
data_dict = load_ssvep_data(subject=1, data_directory='SsvepData/')

#%% Part 2: Plot the Data

# plot the EEG data from electodes Fz and Oz for subject 1
plot_raw_data(data=data_dict, subject=1, channels_to_plot=['Fz','Oz'])

#%% Part 3: Extract the Epochs

# extract epochs (as written for subject 1)
eeg_epochs, epoch_times, is_trial_15Hz = epoch_ssvep_data(data_dict, epoch_start_time=0, epoch_end_time=20)

#%% Part 4: Take the Fourier Transform

# get the frequency spectrum (as written for subject 1)
eeg_epochs_fft, fft_frequencies = get_frequency_spectrum(eeg_epochs, fs=data_dict['fs'])

#%% Part 5: Plot the Power Spectra

# plot the power spectra for electrodes Fz and Oz for subject 1
spectrum_db_15Hz, spectrum_db_12Hz = plot_power_spectrum(eeg_epochs_fft, fft_frequencies, is_trial_15Hz, channels=data_dict['channels'], channels_to_plot=['Fz','Oz'], subject=1)

#%% Run everything for S2

# data_dict2 = load_ssvep_data(subject=2, data_directory='SsvepData/')

# plot_raw_data(data=data_dict2, subject=2, channels_to_plot=['Fz','Oz'])

# eeg_epochs2, epoch_times2, is_trial_15Hz2 = epoch_ssvep_data(data_dict2, epoch_start_time=0, epoch_end_time=20)

# eeg_epochs_fft2, fft_frequencies2 = get_frequency_spectrum(eeg_epochs2, fs=data_dict2['fs'])

#spectrum_db_15Hz2, spectrum_db_12Hz2 = plot_power_spectrum(eeg_epochs_fft2, fft_frequencies2, is_trial_15Hz2, channels=data_dict2['channels'], channels_to_plot=['Fz','Oz'], subject=2)

#%% Part 6: Reflect

"""
1. On some electrodes, there are peaks in the spectra at 12Hz for 12Hz trials and 15Hz for 15Hz trials. What is the name for the brain signal that leads to these peaks? Where in the brain do they originate and why (i.e., what function does this brain region serve)?
    
    a. Steady-state visual evoked potentials (SSVEPs) are the signals that provide the peaks in those spectra [1]. Steady-state refers to the repetitive nature of the flashes presented to the BCI user, and visual evoked potentials are voltages (brain signals measured using electroencephalography) that appear in the signal due to a visual external stimulus. Generally, these peaks are observed within the occipital lobe, which is largely responsible for processign visual data [1].

2. There are smaller peaks at integer multiples of these peak frequencies. What do you call these integer multiples? Why might we see them in the data?
    
    a. These peaks at multiples of the event frequencies are known as harmonics. They occur due to neuronal responses to stimuli, as the neurons respond in a nonlinear fashion [1].

3. There’s a peak in the spectra at 50Hz. What is the most likely cause for this peak?
    
    a. This peak at 50Hz is very likely to be an artifact related to utility frequency. In many regions, 50Hz is the frequency of the alternating current electrical supply, and this electricity can interfere with the signal detected by the electrodes [2]. This signal is often detected as a consequence of poor electrode connection to the scalp [3]; this occurrence is perhaps intuitive given the electrodes at which this 50Hz peak is observed, as the peak is most commonly found at more central electode locations, suggesting factors like hair growth patterns or concentration may prevent ideal connection between the scalp and electode.

4. Besides the sharp peaks just described, there’s a frequency spectrum where the power is roughly proportional to 1 over the frequency. This is a common pattern in biological signals and is sometimes called “1/f” or “neural noise” in EEG. But on top of that, there’s a slight upward bump around 10 Hz on some electrodes. What brain signal might cause this extra power at about 10Hz? Which channels is it most prominently observed on, and why might this be?
    
    a. It is likely that this slight increase around 10Hz is due to alpha wave activity. Alpha brain waves are commonly associated with being in a relaxed state and passive attention [4], which is likely the state in which the participant is experiencing the flashes within the trial. Moreover, the frequency of these waves tends to occur between 8-12Hz [4], further supporting this assertion given the graphical evidence: The "bump" appears to occur around a range of frequencies (around 10Hz) rather than strictly at 10Hz, suggesting the presence of a signal that exhibits a characteristic range of frequencies. The alpha band is interestingly most notable where the 50Hz signal (supposedly interference) is prevalent, though this is not a hard rule.
    Just under half of the electrodes exhibited the 10Hz bump for either or both subjects: Fz, F4, FC1, C3, Cz, C4, CP1, CP2, P3, Pz, P4, O1, Oz, and O2 demonstrated the increased prevalence of signals with roughly 10Hz frequency. These electrodes are all more centralized. Although the bump can be observed for those electrodes listed above, they are most prevalent posteriorly (especially within the occipital lobe), which is consistent with where alpha waves are generally found [5].
    
References:
    [1] D. Jangraw, “13_SSVEPs,” University of Vermont, Feb. 27, 2024.
    [2] A. de Cheveigné, “ZapLine: A simple and effective method to remove power line artifacts,” NeuroImage, vol. 207, p. 116356, Feb. 2020, doi: 10.1016/j.neuroimage.2019.116356.
    [3] “Signal Quality - Recognize and achieve a good EEG signal,” brain-trainer.com. Accessed: Mar. 01, 2024. [Online]. Available: https://brain-trainer.com/support/resources/signal-quality-recognize-and-achieve-good-eeg-signal/
    [4] “Brain Waves - an overview | ScienceDirect Topics.” Accessed: Mar. 01, 2024. [Online]. Available: https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/brain-waves
    [5] “Alpha Wave - an overview | ScienceDirect Topics.” Accessed: Mar. 02, 2024. [Online]. Available: https://www.sciencedirect.com/topics/neuroscience/alpha-wave#:~:text=Alpha%20waves%20can%20be%20found,also%20occurs%20during%20REM%20sleep.

"""