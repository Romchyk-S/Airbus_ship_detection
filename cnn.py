# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:06:28 2024

@author: romas
"""

import tensorflow.keras.layers as tkl

import tensorflow.keras.losses as tklosses

import tensorflow.keras.models as tkm

import time as tm


# import sklearn.model_selection as skms

def main_cnn(X_train, Y_train, kernel_size, pool_size, epochs = 10, batch_size = 5):
    
    model = tkm.Sequential([    
            
        tkl.Rescaling(1./255),
        
          tkl.Conv2D(32, kernel_size = kernel_size, activation='relu', input_shape=X_train.shape[1:4]),
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
         
          tkl.Dense(2)])
        
    model.compile(loss = tklosses.SparseCategoricalCrossentropy(from_logits = True), optimizer='adam', metrics=['accuracy'])


    print("Training the model")

    start = tm.perf_counter()

    model.fit_generator() # train the CNN
    
    print(f"Time to train {round(tm.perf_counter() - start, 3)} seconds")