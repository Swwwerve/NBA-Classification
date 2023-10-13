import json
import numpy as np
import pandas as pd
import cv2
import matplotlib 
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
import pywt

# Load dictionary
with open("celeb_file_names.json", "r") as file:
    celeb_file_name_dict = json.load(file)

# Step 3 - Wavelet transformed images
# We can use wavelet transform for feature engineering --> similar to fourier transform
# Stackoverflow 
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
class_dict = {}
count = 0
for celebrity_name in celeb_file_name_dict.keys():
    class_dict[celebrity_name] = count
    count += 1

# Get training and test sets
X = [] # All the images
Y = [] # A number representing which NBA player it is (class_dict above)
for celebrity_name, training_files in celeb_file_name_dict.items(): # Iterate through every person
    for training_image in training_files: # Iterate through every image for every person
        img = cv2.imread(training_image)
        if img is None: # Since we manually removed images the result of cv2.imread() will be None for some of them
            continue
        scaled_raw_img = cv2.resize(img, (32,32)) # Scaled raw image
        img_har = w2d(img, 'db1', 5)
        scaled_har_img = cv2.resize(img_har, (32,32))
        combined_img = np.vstack((scaled_raw_img.reshape(32*32*3,1),scaled_har_img.reshape(32*32,1))) # 32*32*3 the 3 is for rgb
        X.append(combined_img)
        Y.append(class_dict[celebrity_name])

X = np.array(X).reshape(len(X),4096).astype(float)

# Training model
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf',C=10))]) # Scaling X_train using standardscaler and creating SVM model in 2nd step
pipe.fit(X_train, y_train)

# Using GridSearchCV to hypertune parameters (finding the perfect parameters)
# Also useful in trying out 3 diff. type of models --> random forest, logistic regression, and SVM
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# All models we want to try
# Courtesy of codebasics
model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto',probability=True),
        'params' : {
            'svc__C': [0.1,1,5,10,100,1000],
            'svc__kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'randomforestclassifier__n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'logisticregression__C': [1,5,10,15,20,25]
        }
    }
}

# Running GridSearchCV
# codebasics 
scores = []
best_estimators = {}
import pandas as pd
for algo, mp in model_params.items():
    pipe = make_pipeline(StandardScaler(), mp['model'])
    clf = GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False) 
    clf.fit(X_train, y_train)
    scores.append({
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimators[algo] = clf.best_estimator_
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])

# Finding best model
best_estimators['svm'].score(X_test,y_test)
best_estimators['random_forest'].score(X_test,y_test)
best_estimators['logistic_regression'].score(X_test,y_test)

# Best model
best_clf = best_estimators['logistic_regression']

# Evaluating Model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, best_clf.predict(X_test))

# Step 5 - Save model 
import joblib 
joblib.dump(best_clf, 'saved_model.pkl') 

# Saving class dictionary
with open("class_dictionary.json","w") as f:
    f.write(json.dumps(class_dict))