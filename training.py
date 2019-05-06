import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.createLBPHFaceRecognizer()
path='dataSet'

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    faces=[]
    Ids=[]
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        ID=int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        print ID
        Ids.append(ID)
        cv2.imshow("training",imageNp)
        cv2.waitKey(10)  
    return faces,Ids

faces,Ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(Ids))
recognizer.save('trainner/trainner.yml')
cv2.destroyAllWindows()
