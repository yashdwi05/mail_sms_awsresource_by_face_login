#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#croping face using face detection
def face_extractor(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_classifier.detectMultiScale(gray_img, 1.2, 6)
    if face is():
        return None
    for(x,y,w,h) in face:
        c_face = img[y:y+h, x:x+w]
        return c_face


#resize and save gray images for model training
user = input("Enter UserName:")
cap = cv2.VideoCapture(0)
count = 0
while True:
    ret, photo = cap.read()
    if face_extractor(photo) is not None:
        count+=1
        face = cv2.resize(face_extractor(photo), (200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        file_save = './facedata/' + user + '/' + str(count) + '.jpg'
        cv2.imwrite(file_save, face)
        cv2.putText(face, str(count), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2 )
        cv2.imshow('Faces', face)
    else:
        print('Not Found, Detacting Again...')
        pass
    if cv2.waitKey(10)==13 or count == 100:
        break
cap.release()
cv2.destroyAllWindows()
print('Samples Collected')

