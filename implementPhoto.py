#This works.
#The photo quality means everything. Try to have the head against a white background, with
#photographic exposure. 

import cv2
import numpy as np


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
fontColor = (255, 255, 255)
fontScale = 1

#cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
test_img1 = cv2.imread("testData/andrew2.jpg")
#ret, im =cam.read()
im = test_img1

gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(gray, 1.2,5)

for(x,y,w,h) in faces:
    cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
    Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
    #if(conf&lt>50):
    if(Id==1):
        Id="Andrew"
    elif(Id==2):
        Id="Mom"

    # else:
    #     Id="Unknown"
    cv2.putText(im, str(Id), (x,y+h),font, fontScale, fontColor)

cv2.imshow('image',im)

# if cv2.waitKey(10) & 0xFF==ord('q'):
#     break
#cam.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
