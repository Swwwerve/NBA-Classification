# Classifying NBA athletes

# Imports
import numpy as np
import cv2
import matplotlib 
matplotlib.use('TkAgg',force=True) # Stackoverflow
from matplotlib import pyplot as plt
import os
import shutil
import json

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
            return roi_color # Doesn't account for 2 faces 
        
# If you run plt.imshow() and it result in error, then your image does not have two eyes

# Create folder for all cropped images
path_to_data = r".\Model\dataset"
path_to_crop_data = r".\Model\dataset\cropped"

img_dirs = []
for entry in os.scandir(path_to_data): # Go through all subdirectories in dataset folder
    if entry.is_dir():
        img_dirs.append(entry.path) # all the subdirectories per player
        
img_dirs = img_dirs[1:] # Delete cropped folder
        
# Creating cropped folder if it already doesn't exist
if os.path.exists(path_to_crop_data): # Does the folder already exist?
    shutil.rmtree(path_to_crop_data) # Remove it 
os.mkdir(path_to_crop_data)

# Iterating through images
cropped_image_dirs = [] # Cropped folder path for all 5 players
celebrity_file_names_dict = {} 

for img_dir in img_dirs:
    count = 1
    celebrity_name = img_dir.split('\\')[-1] # Splits directory and celeb_name into two components
    print(celebrity_name)
    
    celebrity_file_names_dict[celebrity_name] = [] # blank array for all image paths in dict
    
    for entry in os.scandir(img_dir): # Go through images in a single player's subdirectory
        roi_color = get_cropped_image_if_2_eyes(entry.path) # roi_color is None if no eyes detected
        if roi_color is not None:
            cropped_folder = path_to_crop_data + "\\" + celebrity_name # Subdirectory within cropped folder per player
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                cropped_image_dirs.append(cropped_folder) # Helper variable
                print("Generating cropped images in folder: ", cropped_folder)
                
            cropped_file_name = celebrity_name + str(count) + ".png" # e.g. giannis1.png, giannis2.png
            cropped_file_path = cropped_folder + "\\" + cropped_file_name 
            
            cv2.imwrite(cropped_file_path, roi_color) # Saving roi_color in cropped_file_path
            celebrity_file_names_dict[celebrity_name].append(cropped_file_path)
            count += 1
            
# Manually examine cropped folder and delete any unwanted images
celebrity_file_names_dict = {}
for img_dir in cropped_image_dirs:
    celebrity_name = img_dir.split('\\')[-1]
    file_list = []
    for entry in os.scandir(img_dir):
        file_list.append(entry.path)
    celebrity_file_names_dict[celebrity_name] = file_list

# Step 2 - manual data cleaning 
for celeb_name, image_list in celebrity_file_names_dict.items():
    for image in image_list:
        image_split = image.split('\\')[-1]
        if celeb_name == 'giannis':
            if image_split == 'giannis18.png' or image_split == 'giannis20.png' or image_split == 'giannis25.png' or image_split == 'giannis33.png' or image_split == 'giannis34.png' or image_split == 'giannis37.png':
                image_list.remove(image)
        elif celeb_name == 'kevin':
            if image_split == 'kevin11.png' or image_split == 'kevin26.png' or image_split == 'kevin35.png':
                image_list.remove(image)
        elif celeb_name == 'lebron':
            if image_split == 'lebron13.png' or image_split == 'lebron17.png' or image_split == 'lebron24.png' or image_split == 'lebron26.png':
                image_list.remove(image)
        elif celeb_name == 'russell':
            if image_split == 'russell5.png' or image_split == 'russell9.png' or image_split == 'russell24.png' or image_split == 'russell28.png':
                image_list.remove(image)
        else:
            if image_split == 'stephen2.png' or image_split == 'stephen12.png' or image_split == 'stephen21.png':
                image_list.remove(image)
        
        celebrity_file_names_dict[celeb_name] = image_list
        
# Put dictionary into new file 
with open("celeb_file_names.json", "w") as file:
    json.dump(celebrity_file_names_dict, file)