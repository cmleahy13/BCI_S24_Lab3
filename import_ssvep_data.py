#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 22:00:05 2024

@author: Peijin Chen and Claire Leahy

Sources:
    
    Avoiding printing "dict_keys" data type: https://blog.finxter.com/python-print-dictionary-keys-without-dict_keys/
    
    Setting axis tickmarks: ChatGPT
    
"""

# import statements
import numpy as np
from matplotlib import pyplot as plt
import scipy.fft

#%% Part 1: Load the Data
def load_ssvep_data(subject, data_directory='SsvepData/'):
    
    # load data
    data = np.load(f'{data_directory}SSVEP_S{subject}.npz', allow_pickle=True)
    
    # create data dictionary (convert existing data type to dict)
    data_dict = {'eeg': data['eeg'], 'channels': data['channels'], 'fs': data['fs'], 'event_samples': data['event_samples'], 'event_durations': data['event_durations'], 'event_types': data['event_types']}
    
    # printing to inform user of some data features
    print(f'Data keys: {list(data_dict.keys())}') # list conversion prevents printing data type
    print('\nChannels: ', data['channels'])
    print('\nSampling frequency (Hz):', data['fs'])
    
    # return data dictionary
    return data_dict

#%% Part 2: Plot the Data

def plot_raw_data(data, subject, channels_to_plot):
    
    # extract data
    eeg = data['eeg']
    channels = list(data['channels'])
    fs = data['fs']
    event_durations = data['event_durations']
    event_samples = data['event_samples']
    event_types = data['event_types']
    
    # find sample times
    t = [] # declare list to contain each sample time in seconds
    # load in times in seconds for all samples
    for sample_index in range(len(eeg.T)):
        
        t.append(sample_index*(1/fs)) # add to time list by dividing sample number by sample frequency
    
    # find event samples and times
    event_intervals = [] # list to contain samples for event intervals
    # find event intervals using corresponding sample index and duration
    for sample_index in range(len(event_samples)):
      
        event_start = event_samples[sample_index]
        event_finish = int(event_samples[sample_index]+event_durations[sample_index])
        
        event_intervals.append((event_start, event_finish))
        
    # times in seconds for event samples
    for sample_index in range(len(event_samples)):
        
        event_intervals[sample_index] = np.array(event_intervals[sample_index])/fs # convert each event sample number to a time, must convert to array first
    
    # initialize figure
    figure, sub_figure = plt.subplots(2, sharex=True)
    
    # top subplot containing flash frequency over span of event
    for event_number, interval in enumerate(event_intervals):
    
        if event_types[event_number] == "12hz":
            sub_figure[0].hlines(xmin = interval[0], xmax = interval[1], y = 12,label = 'o')
            sub_figure[0].plot([interval[0], interval[1]], [12,12], 'bo')
    
        else:
            sub_figure[0].hlines(xmin = interval[0], xmax = interval[1], y = 15,label = 'o')
            sub_figure[0].plot([interval[0], interval[1]], [15,15], 'bo')
    
    # bottom subplot contain raw data from specified electrodes
    for channel_number, channel_name in enumerate(channels_to_plot):
      
        channel_index = channels.index(channel_name) # index of channel of interest considering all channels
        
        eeg_data = data['eeg'][channel_index] # EEG data for the channel
    
        sub_figure[1].plot(t, eeg_data*(10**6), label=channel_name) # plot EEG data in µV from channel
    
    # format figure
    # subplot 1
    sub_figure[0].set_xlabel('time(s)')
    sub_figure[0].set_ylabel('Flash frequency')
    sub_figure[0].set_yticks([12,15])
    sub_figure[0].set_yticklabels(['12Hz','15Hz'])
    sub_figure[0].grid()
    
    # subplot 2
    sub_figure[1].set_xlabel('time (s)')
    sub_figure[1].set_ylabel('Voltage (µV)')
    sub_figure[1].legend(loc='best') # place legend in best location given data
    sub_figure[1].grid()
    
    # set plot title
    figure.suptitle(f'SSVEP Subject {subject} Raw Data')
    
    # general appearance
    figure.tight_layout()
    
    # save image
    plt.savefig(f'SSVEP_S{subject}_rawdata.png')

#%% Part 3: Extract the Epochs

def epoch_ssvep_data(data_dict, epoch_start_time=0, epoch_end_time=20):
  
    # extract data
    eeg = data_dict['eeg']
    channels = list(data_dict['channels']) # convert to list
    fs = data_dict['fs']
    event_durations = data_dict['event_durations']
    event_samples = data_dict['event_samples']
    event_types = data_dict['event_types']
    time_per_epoch = int(fs*(epoch_end_time-epoch_start_time)) # convert to int
    
    list_of_channel_epoch_arrays = []
    
    for channel in channels:
        
        channel_index = channels.index(channel)
        eeg_data = eeg[channel_index]
        epoch_list = [] 
        
        for sample_index in range(len(event_samples)):
            start = event_samples[sample_index]
            end = event_samples[sample_index] + int(event_durations[sample_index])
            epoch_index = eeg_data[start:end]
            epoch_list.append(epoch_index)
    
        epoch_array = np.stack(epoch_list)
    
        list_of_channel_epoch_arrays.append(epoch_array)
   
    eeg_epochs = np.stack(list_of_channel_epoch_arrays, axis=1)
    
    epoch_times = np.linspace(epoch_start_time, epoch_end_time, time_per_epoch)
    
    is_trial_15Hz = [True if event == '15hz' else False for event in event_types]
    
    is_trial_15Hz = np.array(is_trial_15Hz)
    
    return eeg_epochs, epoch_times, is_trial_15Hz

#%% Part 4: Take the Fourier Transform

def get_frequency_spectrum(eeg_epochs, fs):
  
    # take the Fourier Transform of the epoched EEG data
    eeg_epochs_fft = np.fft.rfft(eeg_epochs)
    
    # find the corresponding frequencies from the epoched EEG data
    fft_frequencies = np.fft.rfftfreq(n=eeg_epochs.shape[-1], d=1/fs) # n is the number of samples in the signal (dimension 2 (final dimension) in eeg_epochs), d is the inverse of sampling frequency
  
    return eeg_epochs_fft, fft_frequencies

#%% Part 5: Plot the Power Spectra

def plot_power_spectrum(eeg_epochs_fft, fft_frequencies, is_trial_15Hz, channels, channels_to_plot, subject):

    # convert channels to list
    channels = list(channels)
    
    # calculating power spectra
    # isolate frequency spectra by event type (12Hz or 15Hz)
    event_15 = eeg_epochs_fft[is_trial_15Hz]
    event_12 = eeg_epochs_fft[~is_trial_15Hz]
    
    # calculate power for event type
    event_15_power = (np.abs(event_15))**2
    event_12_power = (np.abs(event_12))**2 
    
    # calculate mean power for event type
    event_15_power_mean = event_15_power.mean(0)
    event_12_power_mean = event_12_power.mean(0)
    
    # calculate normalized power for event type
    normalized_event_15_power_mean = event_15_power_mean/(np.max(event_15_power_mean))
    normalized_event_12_power_mean = event_12_power_mean/(np.max(event_12_power_mean))
    
    # calculate spectra for event type
    spectrum_db_15Hz = 10*(np.log10(normalized_event_15_power_mean))
    spectrum_db_12Hz = 10*(np.log10(normalized_event_12_power_mean))
    
    # plotting
    # isolate channel being plotted
    channel_to_plot = [channels.index(channel_name) for channel_name in channels_to_plot]
    
    # set up figure
    figure, channel_plot = plt.subplots(len(channels_to_plot))
    
    for plot_index, channel in enumerate(channel_to_plot): # plot_index as a means of accessing a subplot
        
        # plot the power spectra by event type
        channel_plot[plot_index].plot(fft_frequencies, spectrum_db_12Hz[channel,:], color='red')
        channel_plot[plot_index].plot(fft_frequencies, spectrum_db_15Hz[channel,:], color='green')
        
        # formatting subplot
        channel_plot[plot_index].set_xlim(0,80)
        channel_plot[plot_index].set_xlabel('frequency (Hz)')
        channel_plot[plot_index].set_ylabel('power (dB)')
        channel_plot[plot_index].set_title(f'Channel {channels_to_plot[plot_index]}')
        channel_plot[plot_index].legend(['12Hz','15Hz'], loc='best')
        channel_plot[plot_index].grid()
        
        # plot dotted lines at 12Hz and 15Hz
        channel_plot[plot_index].axvline(12, color='red', linestyle='dotted')
        channel_plot[plot_index].axvline(15, color='green', linestyle='dotted')
    
    # format overall plot
    figure.suptitle(f'SSVEP Subject S{subject} Frequency Content')
    figure.tight_layout()
    
    # save image
    plt.savefig(f'SSVEP_S{subject}_frequency_content.png')
    
    return spectrum_db_15Hz, spectrum_db_12Hz