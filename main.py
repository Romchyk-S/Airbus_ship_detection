# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 10:37:48 2024

@author: romas
"""

import pandas as pd

import cv2 as cv2

# import matplotlib.pyplot as plt

# import numpy as np

import helper_functions as hf

import data_generator_class as dgc

import cnn as cnn

import unet as unet



train_folder_path = './train_v2'

train_csv = "train_ship_segmentations_v2"

try:
    
    data = pd.read_csv(f"{train_csv}_aggregated.csv")
    
    print(data.tail())
    
    print()

except FileNotFoundError:
    
    print("Aggregating and saving the data in a new csv")
    
    hf.aggregate_and_save_to_csv(train_csv)
    
epochs, batch_size  = 10, 5

kernel_size_cnn = 2, 2

pool_size_cnn = 2, 2

# cnn.main_cnn(X_train, Y_train, kernel_size, pool_size, epochs, batch_size)


kernel_size_downsample = 3, 3

pool_size_unet = 2, 2

kernel_size_upsample = 2, 2

# unet.main_unet(X_train, Y_train, epochs, batch_size, kernel_size_downsample,
#          kernel_size_upsample, pool_size_unet)