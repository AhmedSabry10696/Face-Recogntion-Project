import cv2
import sqlite3

cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

Id=raw_input('enter your id: ')
sampleNum=0
while(True): 
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        sampleNum+=1
        cv2.imwrite("dataSet/User."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) 
        cv2.waitKey(100) 
    cv2.imshow('face',img)
    cv2.waitKey(1) 
    if sampleNum>20:
        break
cam.release()
cv2.destroyAllWindows()
