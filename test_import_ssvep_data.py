#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 22:02:31 2024

@author: Peijin Chen and Claire Leahy
"""

# import statements
from import_ssvep_data import load_ssvep_data, plot_raw_data

#%% Part 1: Load the Data

# load data for subject 1
data_dict = load_ssvep_data(subject=1, data_directory='SsvepData/')

#%% Part 2: Plot the Data

# plot the EEG data from electodes Fz and Oz for subject 1
plot_raw_data(data=data_dict, subject=1, channels_to_plot=['Fz','Oz'])

#%% Part 6: Reflect