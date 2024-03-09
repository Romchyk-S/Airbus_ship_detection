# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:36:38 2024

@author: romas
"""

import keras as keras

class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, IDs, Y, 
                 batch_size = 32, dims=(192555,768,768), channels=3,
                 classes=2, shuffle=True):

        print(batch_size)    