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
    
    #extract data
    event_durations = data['event_durations']
    event_samples = data['event_samples']
    event_types = data['event_types']
    all_channels = list(data['channels'])
    
    #compute event intervals and start/finish times
    event_intervals = []
    for sample_index in range(len(event_samples)):
      
        event_start = event_samples[sample_index]
        event_finish = int(event_samples[sample_index] + event_durations[sample_index])
        
        event_intervals.append((event_start, event_finish))
    
    # define time axis
    t = np.arange(len(data['eeg'].T)) # tentative, number of samples
    
    # initialize figure
    fig, ax = plt.subplots(2, sharex=True)
    
    # top subplot to contain event start/end times, types
    
    # set plot title
    ax[0].set_title(f'SSVEP Subject {subject} Raw Data')
    
    #subplot 1
    for event_num, interval in enumerate(event_intervals):
    
        if event_types[event_num] == "12hz":
            ax[0].hlines(xmin = interval[0], xmax = interval[1], y = 12,label = 'o')
            ax[0].plot([interval[0], interval[1]], [12,12], 'bo')
    
        else:
            ax[0].hlines(xmin = interval[0], xmax = interval[1], y = 15,label = 'o')
            ax[0].plot([interval[0], interval[1]], [15,15], 'bo')
    
    ax[0].set_xlabel('time(s)')
    ax[0].grid(alpha=0.2,color="black")
    #ax[0].vlines([x if x % 1000 == 0 else None for x in range(data['eeg'].shape[-1])], ymin = 12, ymax = 15)
      
    # bottom subplot to contain raw data from specified electrodes
    for channel_number, channel_name in enumerate(channels_to_plot):
      
        channel_index = all_channels.index(channel_name) # index of channel of interest considering all channels
        
        eeg_data = data['eeg'][channel_index] # define EEG data for the channel
    
        ax[1].plot(t, eeg_data*(10**6)) # plot EEG data in uV from given channel
        #ax[1].set_title(f"Channel {channel} EEG plot")
    
    # format figure
    fig.tight_layout()
    ax[1].set_xlabel('time (s)')
    
    # save image
    plt.savefig(f'SSVEP_S{subject}_rawdata.png')

#%% Part 3: Extract the Epochs
def epoch_ssvep_data(data_dict, epoch_start_time = 0, epoch_end_time = 20):
  
  event_samples = data_dict['event_samples']
  event_types = data_dict['event_types']
  event_durations = data_dict['event_durations']
  all_channels = list(data_dict['channels'])
  
  list_of_channel_epoch_arrays = [] 
  
  for channel in all_channels:
    channel_idx = all_channels.index(channel)
    eeg_data = data['eeg'][channel_idx]
    epoch_list = [] 
    for i in range(len(event_samples)):
      start = event_samples[i]
      end = event_samples[i] + int(event_durations[i])
      #print(start,end)
      epoch_i = eeg_data[start:end]
      epoch_list.append(epoch_i)

    epoch_array = np.stack(epoch_list)
    #epoch_array = np.reshape(epoch_array, (20,1,20000))
    list_of_channel_epoch_arrays.append(epoch_array)
  eeg_epochs = np.stack(list_of_channel_epoch_arrays, axis = 1)
  epoch_times = np.linspace(epoch_start_time, epoch_end_time, int(data_dict['fs'] * (epoch_end_time - epoch_start_time)))
  is_trial_15Hz = [True if x == '15hz' else False for x in event_types] 
  return eeg_epochs, epoch_times, np.array(is_trial_15Hz)

#%% Part 4: Take the Fourier Transform


def get_frequency_spectrum(eeg_epochs, fs):
  eeg_epochs_fft = np.fft.rfft(eeg_epochs)
  fft_frequencies = np.fft.rfftfreq(n = eeg_epochs.shape[-1], d = 1/fs)
  return eeg_epochs_fft, fft_frequencies

#%% Part 5: Plot the Power Spectra

def plot_power_spectrum(eeg_epochs_fft, fft_frequencies, is_trial_15hz, channels, channels_to_plot, subject):

  channels = list(channels)
  a15 = eeg_epochs_fft[is_trial_15hz]
  a12 = eeg_epochs_fft[~is_trial_15hz]
  

  a15_power = np.abs(a15) ** 2
  a12_power = np.abs(a12) ** 2 

  a15_power_mean_across_trials = a15_power.mean(0)
  a12_power_mean_across_trials = a12_power.mean(0)
  normalized_a15_power_mean_across_trials = a15_power_mean_across_trials / np.max(a15_power_mean_across_trials)
  normalized_a12_power_mean_across_trials = a12_power_mean_across_trials / np.max(a12_power_mean_across_trials)
  spectrum_db_15Hz = 10 * np.log10(normalized_a15_power_mean_across_trials)
  spectrum_db_12Hz = 10 * np.log10(normalized_a12_power_mean_across_trials)

  

  channel_to_plot_idx = [channels.index(channel)for channel in channels_to_plot]

  fig, ax = plt.subplots(len(channels_to_plot))
  for idx, num in enumerate(channel_to_plot_idx):
    ax[idx].plot(fft_frequencies, spectrum_db_15Hz[num,:])
    ax[idx].plot(fft_frequencies, spectrum_db_12Hz[num,:])
    ax[idx].set_xlim(0,80)
    ax[idx].set_title(channels_to_plot[idx] + " frequency content for SSVEP S" + str(subject))
    ax[idx].legend(["15Hz","12Hz"])
  fig.tight_layout()
  plt.show()
  return spectrum_db_15Hz, spectrum_db_12Hz



