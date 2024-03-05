# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 10:37:48 2024

@author: romas
"""

import pandas as pd

# import cv2 as cv2

# import matplotlib.pyplot as plt

# import numpy as np

# import os as os

# import utils as utils

import time as tm

# from PIL import ImageOps

# from datasets import load_dataset

# import tensorflow

import helper_functions as hf

import data_generator_class as dgc

import cnn as cnn

import unet as unet


kernel_size = 3,3

pool_size = 2,2

data = pd.read_csv("train_ship_segmentations_v2.csv")

train_folder_path = './train_v2'

data_1 = data.copy()


data_1['Ship_exists'] = data_1['EncodedPixels'].notnull()

data_1['EncodedPixels'] = data['EncodedPixels']

start = tm.perf_counter()

data_1 = data_1.groupby('ImageId').agg({'ImageId': 'first', 'Ship_exists': ['first', 'sum'], 
                                        'EncodedPixels': lambda rle_code: hf.group_encoded_pixels(rle_code)}) 

data_1.columns = ['ImageId', 'Ship_exists', 'Number_of_ships', 'EncodedPixels_agg']

print(data_1.head())

print()

kernel_size = 2, 2

pool_size = 2, 2

epochs = 10

batch_size = 5

# cnn.main_cnn(X_train, Y_train, kernel_size, pool_size, epochs, batch_size)


kernel_size_downsample = 3, 3

pool_size = 2, 2

kernel_size_upsample = 2, 2

# unet.main_unet(X_train, Y_train, epochs, batch_size, kernel_size_downsample,
#          kernel_size_upsample, pool_size)