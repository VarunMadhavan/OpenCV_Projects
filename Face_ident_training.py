import os
import cv2 as cv
import numpy as np

# labels of training sets
p = {}
for i,name in enumerate(os.listdir('Resources/Faces/train')):
    p[i] = name

train_dir = 'Resources/Faces/train'
def create_train_set():
    # importing haar-cascade classifier weights
    haar_cascade = cv.CascadeClassifier('haar_face.xml')

    features = []
    labels = []

    for i, name in p.items():
        path = os.path.join(train_dir, name)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            img_arr = cv.imread(img_path)
            grayed = cv.cvtColor(img_arr, cv.COLOR_BGR2GRAY)

            rects = haar_cascade.detectMultiScale(grayed, scaleFactor=1.1, minNeighbors=5)
            for x,y,w,h in rects:
                roi = grayed[y:y+h, x:x+w]
                features.append(roi)
                labels.append(i)
    return features, labels
features, labels = create_train_set()
# print(f' features = {len(features)}, lables = {len(labels)}')
print("... Training set created...")

face_recognizer = cv.face.LBPHFaceRecognizer_create()
# Train recognizer
features = np.array(features, dtype='object')
labels = np.array(labels)
face_recognizer.train(features, labels)
print("... Recognizer trained...")

# Saving info
np.save('features.npy', features)
np.save('labels.npy', labels)
face_recognizer.save('face_recognizer.yml')
print("... Saving information")








