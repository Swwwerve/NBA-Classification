import cropped_images
import numpy as np
import cv2
import matplotlib 
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
import os
import shutil
import pywt

# Step 3 - Wavelet transformed images
# We can use wavelet transform for feature engineering --> similar to fourier transform
def w2d(img, mode='haar', level=1):
    imArray = img
    # Datatype conversions
    # Convert to grayscale 
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    # Convert to float
    imArray = np.float32(imArray)
    imArray /= 255
    # Compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    
    # Process coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    
    # Reconstruction
    imArray_H = pywt.waverec2(coeffs_H,mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    
    return imArray_H

# Step 4 - Train model 
# Step 5 - Save model 