# Classifying NBA athletes

# Imports
import numpy as np
import cv2
import matplotlib 
from matplotlib import pyplot as plt

# Step 1 - crop faces with 2 eyes using haar cascade
# For a sample image
img = cv2.imread(r'.\Model\dataset\giannis\-1x-1.jpg')
img.shape

# Change to gray image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray.shape

# Haar cascade --> An algorithm that can detect objects in images by scanning for a particular 'dark and light' pattern
# It has pre-built classifiers to detect eyes, nose, mouth etc
# Detecting face and eyes
face_cascade = cv2.CascadeClassifier(r'.\Model\opencv\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'.\Model\opencv\haarcascade_eye.xml')

# Gray is the gray image of nba player and the returned array of 4 values shows you x,y,width,height of face
faces = face_cascade.detectMultiScale(gray, 1.3,5)
print(faces)

# Saving coordinates of face
(x,y,w,h) = faces[0]

# Drawing rectangle over face
face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

# Step 2 - manual data cleaning 
# Step 3 - Wavelet transformed images
# Step 4 - Train model 
# Step 5 - Save model 