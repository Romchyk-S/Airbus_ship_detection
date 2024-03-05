# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:08:48 2024

@author: romas
"""

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
        
    # print(mask)
    
    error_file = "error_pixels.txt"
    
    # file_size = os.path.getsize(error_file)
    
    for x, y in mask:
        
        try:
        
            image[x, y, [0, 1, 2]] = color
            
        except IndexError:
            
            print("Found error pixels")
            
            print(x, y)
            
            with open(error_file, "a") as f:
                
                f.write(f"{x}, {y} \n")
                
            break
                
    return image

def group_encoded_pixels(rle_code):
    
    if type(rle_code.values[0]) != float:
        
        return ' '.join(rle_code)
    
    return 0