# Classifying NBA athletes

# Imports
import numpy as np
import cv2
import matplotlib 
from matplotlib import pyplot as plt
import os
import shutil

# Sample image 
'''
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

# Draw two eyes (opencv documentation) 
for (x,y,w,h) in faces: # Only need this loop once per face 
    face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = face_img[y:y+h, x:x+w] # Cropped image with only eyes (we will save this for training)
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
'''

# Step 1 - crop faces with 2 eyes using haar cascade    
face_cascade = cv2.CascadeClassifier(r'.\Model\opencv\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'.\Model\opencv\haarcascade_eye.xml')

def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w] # Regions of interest with starting and ending coords (indexing for an image)
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color
        
# If you run plt.imshow() and it result in error, then your image does not have two eyes

# Create folder for all cropped images
path_to_data = r".\Model\dataset"
path_to_crop_data = r".\Model\dataset\cropped"

img_dirs = []
for entry in os.scandir(path_to_data): # Go through all subdirectories in dataset folder
    if entry.is_dir():
        img_dirs.append(entry.path)
        
# Creating cropped folder if it already doesn't exist
if os.path.exists(path_to_crop_data): # Does the folder already exist?
    shutil.rmtree(path_to_crop_data) # Remove it 
os.mkdir(path_to_crop_data)

# Step 2 - manual data cleaning 
# Step 3 - Wavelet transformed images
# Step 4 - Train model 
# Step 5 - Save model 