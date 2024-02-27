#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 22:00:05 2024

@author: Peijin Chen and Claire Leahy

Sources:
    
    Avoiding printing "dict_keys" data type: https://blog.finxter.com/python-print-dictionary-keys-without-dict_keys/
    
"""

# import statements
import numpy as np
from matplotlib import pyplot as plt
import scipy.fft

#%% Part 1: Load the Data
def load_ssvep_data(subject, data_directory='SsvepData/'):
    
    # load data
    data = np.load(f'{data_directory}/SSVEP_S{subject}.npz', allow_pickle=True)
    
    # create data dictionary
    data_dict = {'eeg': data['eeg'], 'channels': data['channels'], 'fs': data['fs'], 'event_samples': data['event_samples'], 'event_durations': data['event_durations'], 'event_types': data['event_types']}
    
    # printing to inform user of some data features
    print(f'Data keys: {list(data_dict.keys())}') # list conversion prevents printing data type
    print('\nChannels: ', data['channels'])
    print('\nSampling frequency (Hz):', data['fs'])
    
    # return data dictionary
    return data_dict

#%% Part 2: Plot the Data
def plot_raw_data(data, subject, channels_to_plot):
    
    # create a list of all of the channels
    all_channels = list(data['channels']) # indexing error
    
    # define time axis
    t = np.arange(data['eeg'].T) # tentative, number of samples
    
    # initialize figure
    fig, ax = plt.subplots(2)
    
    # top subplot to contain event start/end times, types
    #ax[0].plot(data[''])
    
    # bottom subplot to contain raw data from specified electrodes
    for channel_number, channel_name in enumerate(channels_to_plot):
      
        channel_index = all_channels.index(channel_name) # index of channel of interest considering all channels
        
        eeg_data = data['eeg'][channel_index] # define EEG data for the channel
    
        ax[1].plot(t, eeg_data) # plot EEG data from given channel
        #ax[1].set_title(f"Channel {channel} EEG plot")
    
    fig.tight_layout()
    
    # save image
    plt.savefig(f'SSVEP_S{subject}_rawdata.png')

#%% Part 3: Extract the Epochs

#%% Part 4: Take the Fourier Transform

#%% Part 5: Plot the Power Spectra