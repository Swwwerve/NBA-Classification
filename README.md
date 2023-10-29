# NBA-Classification
This is a project using machine learning (GridSearchCV to find optimal model) to determine whether a given image is one of 5 NBA players (Giannis, Durant, Lebron, Westbrook, Curry)

# Packages
pip install -r requirements.txt

# IMPORTANT NOTE
Flask server is still in development 

# Process
## Dataset
1. Compiled large dataset of images of 5 NBA players using Fatkun batch download google chrome extension
2. Used OpenCV with haarcascade to isolate for face and eyes in particular
3. Manually deleted any images that were not of desired player
4. Cropped images as shown in cropped_images.py file
5. Appended all cropped images to new subdirectory (cropped)

## Model
1. Used python wavelet transform for feature engineering and to convert images into coefficients to ne interpreted by model
2. Used GridSearchCV to search for best classification model (between SVM, random forest, logistic regression) and to hypertune parameters
3. Saved best model (logistic regression) in pkl file

## Server
1. Used Flask module in python to create a basic Flask server with one decorator app route for 'classify_image'
2. Essentially this server would run the saved pkl file on the provided image (image converted to base64 encoded string)

## UI
1. Created basic website using HTML, Bootstrap CSS, JS (communicates with Flask Server)

# Reference
https://www.youtube.com/@codebasics
