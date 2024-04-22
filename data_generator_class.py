# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:36:38 2024

@author: romas
"""

import keras as keras

import numpy as np

import cv2 as cv2

class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, X, Y, set_type,
                 batch_size = 32, dims=(192556,768,768), channels=3,
                 classes=2, shuffle=True):
        
        self.X = X
        
        self.samples = len(self.X)
        
        self.labels = Y
        
        self.set_type = set_type
        
        self.batch_size = batch_size
        
        self.dims = dims
        
        self.channels = channels
        
        self.classes = classes
        
        self.shuffle = shuffle
        
        self.on_epoch_end()
        
    def __len__(self) -> int:
        'batches in epoch'
        
        return int(np.floor(self.samples/self.batch_size))
       
    def __getitem__(self, index):
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        X, Y = self.__data_generation(indexes)
        
        # print(X)
        
        # print(Y)

        return X, Y
    
    def print_batch_info(self):
        
        print(f"Sample_size: {self.samples}")
        
        print(f"Batch size: {self.batch_size}")
        
        print(f"Batches per epoch: {len(self)}")
        
        print()
    
    def on_epoch_end(self):
        
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.samples)
        
        if self.shuffle == True:
            
            np.random.shuffle(self.indexes)
            
            
                
    def __data_generation(self, indexes):
        
        # print("Data generation")
        
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_generated = np.empty((self.batch_size, *self.dims[1:4], self.channels))
        Y_generated = np.empty((self.batch_size))
        
        # print(X_generated.shape)
        
        # print(Y_generated.shape)
        
        # print(indexes)
        
        for i, ID in enumerate(indexes):
          # Store sample, Here may need to load images for unet.
          X_generated[i,] = cv2.imread(f'{self.set_type}_v2/{self.X[ID]}')

          # Store class
          Y_generated[i] = self.labels[ID]
          
        # print(X_generated.shape)
        
        # print(Y_generated)
        
        # print()
        
        return X_generated, keras.utils.to_categorical(Y_generated, num_classes=self.classes)