import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
rec= cv2.createLBPHFaceRecognizer()
rec.load('trainner/trainner.yml')
id=0
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1,1,0,4,1) 
while True:
    ret,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
        id,conf = rec.predict(gray[y:y+h,x:x+w])
        print id
        print conf
        if(id==1):
            id="Ahmed"
        else:
            id="not known"
        cv2.cv.PutText(cv2.cv.fromarray(img),str(id), (x,y+h),font, 255)
    cv2.imshow('img',img)
 
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
    
cap.release()
cv2.destroyAllWindows()
