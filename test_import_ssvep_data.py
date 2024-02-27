#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 22:02:31 2024

@author: Peijin Chen and Claire Leahy
"""

# import statements
from import_ssvep_data import load_ssvep_data

#%% Part 1: Load the Data

# load data for subject 1
data_dict = load_ssvep_data(1, data_directory='SsvepData/')

#%% Part 6: Reflect