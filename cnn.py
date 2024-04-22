# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:06:28 2024

@author: romas
"""

import tensorflow as tf

import tensorflow.keras.layers as tkl

import tensorflow.keras.losses as tklosses

import tensorflow.keras.models as tkm

import time as tm

import numpy as np


# import sklearn.model_selection as skms

def dice_coef(y_true, y_pred, smooth=1):
    # Reshape the true masks
    # y_true = K.cast(y_true, 'float32')
    # Calculate the intersection between predicted and true masks
    intersection = np.sum(y_true * y_pred, axis=[1, 2, 3])
    # Calculate the union of predicted and true masks
    union = np.sum(y_true, axis=[1, 2, 3]) + np.sum(y_pred, axis=[1, 2, 3])
    # Calculate the Dice coefficient
    return np.mean((2. * intersection + smooth) / (union + smooth), axis=0)

def main_cnn(generator, input_shape, kernel_size, pool_size, epochs = 10, use_GPU = True):
    
    model = tkm.Sequential([    
            
        tkl.Rescaling(1./255),
        
          tkl.Conv2D(32, kernel_size = kernel_size, activation='relu'),
          tkl.MaxPooling2D(pool_size = pool_size),
         
            tkl.Conv2D(64, kernel_size = kernel_size, activation='relu'),
            tkl.MaxPooling2D(pool_size = pool_size),
         
            tkl.Conv2D(128, kernel_size = kernel_size, activation='relu'),
            tkl.MaxPooling2D(pool_size = pool_size),
        
            tkl.Flatten(),
         
          # try and show the inner layer workings.
         
            tkl.Dense(128, activation='relu'),
          
            tkl.Dense(64, activation='relu'),
         
            tkl.Dense(32, activation='relu'),
         
           tkl.Dense(16, activation='relu'),
         
          tkl.Dense(generator.classes)])
        
    model.compile(loss = tklosses.BinaryCrossentropy(from_logits = True), optimizer='adam', metrics=['accuracy'])
    
    # model.compile(loss = dice_coef, optimizer='adam', metrics=['accuracy'])

    print("Training the model")

    start = tm.perf_counter()
        
    if use_GPU:

        model.fit(generator, epochs = epochs)
            
    else:
        
        with tf.device('CPU'):

            model.fit(generator, epochs = epochs)
        
    print(f"Training time is {round(tm.perf_counter() - start, 3)} seconds")