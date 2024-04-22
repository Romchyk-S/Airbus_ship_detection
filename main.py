# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 10:37:48 2024

@author: romas
"""

import pandas as pd

import cv2 as cv2

# import matplotlib.pyplot as plt

import numpy as np

# import dtale as dtale

import helper_functions as hf

import data_generator_class as dgc

import cnn as cnn

import unet as unet

import keras as keras

import tensorflow as tf

import os

import sklearn.model_selection as skms

import torch


# print(torch.cuda.is_available())

# os.environ['CUDA_HOME'] = os.environ.get('CUDA_PATH')

# print(os.environ.get('CUDA_HOME'))

# print(os.environ.get('CUDA_PATH'))

# print(tf.config.list_physical_devices())

if len(tf.config.list_physical_devices('GPU')) > 0:
    
    print("GPU available")
    
    use_GPU = True
    
else:
    
    print("No GPU available.")
    
    use_GPU = False
    
print()



# app = dtale.app.build_app(reaper_on = False)
# dtale.show(data[0:5], host = 'localhost')
# app.run(host = "localhost", port=8888)

with tf.device("CPU"):
    
    tf.random.set_seed(16)

    config = tf.compat.v1.ConfigProto()
    
    config.gpu_options.allow_growth = True
    
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    
    sess = tf.compat.v1.Session(config=config)
    
    train_folder_path = './train_v2'
    
    train_csv = "train_ship_segmentations_v2"
    
    test_csv = "sample_submission_v2"
    
    data_train = hf.read_or_aggregate_data(train_csv)
    
    zero_indices = [ind for ind in data_train[data_train.columns[0]] if data_train['Ship_exists'][ind] == False]
    
    old_data_len = len(data_train)

    
    zero_indices_to_drop = list(np.random.choice(zero_indices, len(zero_indices)-25000, replace=False))
    
    data_train = data_train.drop(zero_indices_to_drop).reset_index()
    
    epochs, batch_size = 1, 20

    kernel_size_cnn = 2, 2
    
    pool_size_cnn = 3, 3
    
    data_train, data_valid = skms.train_test_split(data_train, 
                 test_size = 0.3, 
                 stratify = data_train['Ship_exists'])
    
    data_train = data_train.reset_index()
    
    X_train_cnn = data_train['ImageId']
    
    Y_train_cnn = data_train['Ship_exists']
    
    image_shape = cv2.imread(f"train_v2/{X_train_cnn[0]}").shape

    training_generator = dgc.DataGenerator(X_train_cnn, Y_train_cnn, set_type="train", classes = len(set(Y_train_cnn)), 
                                            dims = (len(X_train_cnn), image_shape[0], image_shape[1]), 
                                            channels = image_shape[2], batch_size = batch_size)
    
    training_generator.print_batch_info()

    cnn.main_cnn(training_generator, image_shape, kernel_size_cnn, pool_size_cnn, epochs, use_GPU = use_GPU)


kernel_size_downsample = 3, 3

pool_size_unet = 2, 2

kernel_size_upsample = 2, 2

# data_train, data_valid = skms.train_test_split(data_train, 
#              test_size = 0.3, 
#              stratify = data_train['NumberOfShips'])


# unet.main_unet(X_train, Y_train, epochs, batch_size, kernel_size_downsample,
#          kernel_size_upsample, pool_size_unet)