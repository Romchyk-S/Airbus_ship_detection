# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:08:48 2024

@author: romas
"""

import pandas as pd

def read_or_aggregate_data(file_name: str):
    
    try:
        
        data = pd.read_csv(f"{file_name}_aggregated.csv")

    except FileNotFoundError:
        
        data_type = file_name.split("_")[0]
        
        print(f"Aggregating and saving the data in a new csv for {data_type}ing.")
        
        aggregate_and_save_to_csv(file_name)
        
        data = pd.read_csv(f"{file_name}_aggregated.csv")
        
    return data

def rle_to_pixels(rle_code, img_shape):
    ''' This function decodes Run Length Encoding into pixels '''
    
    print(rle_code)
    
    rle_code = [int(i) for i in rle_code.split()]

    pixels = [(pixel_position % img_shape[0], pixel_position // img_shape[1]) 
              for start, length in list(zip(rle_code[0:-1:2], rle_code[1::2])) 
              for pixel_position in range(start, start + length)]
    
    print("Checking for faults")
    
    print(len([p for p in pixels if p[0] > img_shape[0] or p[1] > img_shape[1]]))
    
    print()
        
    return pixels

def apply_mask(image, mask, color = (255, 255, 0)):
    
    ''' This function saturates the Red and Green RGB colors in the image 
        where the coordinates match the mask'''
        
    for x, y in mask:
        
        image[x, y, [0, 1, 2]] = color
            
       
    return image

def group_encoded_pixels(rle_code):
    
    if type(rle_code.values[0]) != float:
        
        return ' '.join(rle_code)
    
    return 0

def aggregate_and_save_to_csv(file_name: str):
    
    data = pd.read_csv(f"{file_name}.csv")
    
    data_1 = data.copy()
    
    data_1['Ship_exists'] = data_1['EncodedPixels'].notnull()
    
    data_1['EncodedPixels'] = data['EncodedPixels']
    
    
    data_1 = data_1.groupby('ImageId').agg({'Ship_exists': ['first', 'sum'], 
                                            'EncodedPixels': lambda rle_code: group_encoded_pixels(rle_code)}).reset_index()
    
    data_1.columns = ['ImageId', 'Ship_exists', 'Number_of_ships', 'EncodedPixels_agg']
    
    data_1.to_csv(f'{file_name}_aggregated.csv', sep=',', encoding = 'utf_8')