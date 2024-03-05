# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:06:38 2024

@author: romas
"""

import tensorflow as tf

def conv_relu(inputs, num_filters, kernel_size):
    
    # print(inputs)
    
    x = tf.keras.layers.Conv2D(num_filters, kernel_size = kernel_size, padding = 'valid')(inputs) 
    x = tf.keras.layers.Activation('relu')(x) 
    
    # kernel_size = (num+1 for num in kernel_size)
      
    # Convolution with 3x3 filter followed by ReLU activation 
    x = tf.keras.layers.Conv2D(num_filters, kernel_size = kernel_size+1, padding = 'valid')(x) 
    x = tf.keras.layers.Activation('relu')(x) 

    return x

def encoder(inputs, num_filters, pool_size, kernel_size):
    
    x = conv_relu(inputs, num_filters, kernel_size)
    
    x = tf.keras.layers.MaxPool2D(pool_size = pool_size, strides = 2)(x) 
    
    return x

def decoder(inputs, skip_features, num_filters, kernel_size):
    
    # Upsampling with 2x2 filter 
   x = tf.keras.layers.Conv2DTranspose(num_filters, kernel_size, strides = 2, padding = 'valid')(inputs) 
     
   # Copy and crop the skip features to match the shape of the upsampled input 
   skip_features = tf.image.resize(skip_features, size = (x.shape[1], x.shape[2])) 
   
   x = tf.keras.layers.Concatenate()([x, skip_features]) 
    
   
   x = conv_relu(inputs, num_filters)
    
   return x
    

def main_unet(X_train, Y_train, epochs = 10, batch_size = 5, kernel_size_downsample = (3,3),
         kernel_size_upsample = (2,2), pool_size = (2,2)):
    
    input_shape = X_train.shape[1:4]

    inputs = tf.keras.layers.Input(input_shape) 
    
    inputs = tf.keras.layers.Rescaling(1./255)(inputs)
    
    s1 = encoder(inputs, 64, kernel_size_downsample)
    
    s2 = encoder(s1, 128, kernel_size_downsample)
    
    s3 = encoder(s2, 256 ,kernel_size_downsample)
    
    s4 = encoder(s3, 512, kernel_size_downsample)
    
    b1 = conv_relu(s4, 1024)
    
    s5 = decoder(b1, s4, 512, kernel_size_upsample)
    
    s6 = decoder(s5, s3, 256, kernel_size_upsample)
    
    s7 = decoder(s6, s2, 128, kernel_size_upsample)

    s8 = decoder(s7, s1, 64, kernel_size_upsample)

    outputs = tf.keras.layers.Conv2D(2, 1, padding = 'valid', activation = 'sigmoid')(s8) 
    
    model = tf.keras.models.Model(inputs = inputs, outputs = outputs, name = 'U-Net') 
    
    model.fit(X_train, Y_train, epochs = epochs, batch_size = batch_size)
    
    return model 
