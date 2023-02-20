import cv2 as cv
import os

# labels of training sets
p = {}
for i, name in enumerate(os.listdir('Resources/Faces/train')):
    p[i] = name
def identify_face(img_path, face_recognizer, haar_cascade):
    img = cv.imread(img_path)
    grayed = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rects = haar_cascade.detectMultiScale(grayed, 1.1, 5)
    for x,y,w,h in rects:
        roi = grayed[y:y+h, x:x+w]
        label, conf = face_recognizer.predict(roi)

        cv.putText(img, p[label], (25,25), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=1)
        cv.putText(img, str(conf), (x, y+h+24), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), thickness=1)
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    cv.imshow('Recognized Face', img)

    cv.waitKey(0)
path = 'Resources/Faces/val/madonna/4.jpg'
haar_cascade = cv.CascadeClassifier('haar_face.xml')
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_recognizer.yml')
identify_face(path, face_recognizer, haar_cascade)