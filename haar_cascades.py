import cv2 as cv

# sample image
img = cv.imread('Resources/Photos/group 1.jpg')
cv.imshow('Person', img)

# Graying image
grayed = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Grayed person', grayed)

# importing haar-cascade classifier weights
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Option values
_scaleFactor = 1.1
_minNeighbors = 1

# Detecting faces
faces_rect = haar_cascade.detectMultiScale(grayed, scaleFactor=_scaleFactor, minNeighbors=_minNeighbors)
print(f' Faces found = {len(faces_rect)}')

# Showing faces
for x,y,w,h in faces_rect:
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

cv.imshow('Face', img)

cv.waitKey(0)