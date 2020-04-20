from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matlab

model = load_model('Project Dataset/Object_Recognition_model/my_model_fianal.h5')
#model = matlab.load('C:/Users/Lenovo/Desktop/Project Dataset/Object_Recognition_model/my_model2.h5')

import requests
from PIL import Image

##img = Image.open('four.png')
##
####url = 'https://www.researchgate.net/profile/Jose_Sempere/publication/221258631/figure/fig1/AS:305526891139075@1449854695342/Handwritten-digit-2.png'
####response = requests.get(url, stream=True)
####img = Image.open(response.raw)
##plt.figure(0)
##plt.imshow(img, cmap=plt.get_cmap('gray'))
##
##
##img = np.asarray(img)
##img = cv2.resize(img, (28, 28))
##img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##img = cv2.bitwise_not(img)q
##plt.figure(1)
##plt.imshow(img, cmap=plt.get_cmap('gray'))
##
##img = img/255
##img = img.reshape(1, 784)
##
##prediction = model.predict_classes(img)
##print("predicted digit:", str(prediction))

##plt.show()



objects = ['Blind Stick',
         'Socket',
         'Telephone',
         'Mobile',
         'Carvaan',
         'Himalaya Tablet',
         'Fork',
         'Tissue Box',
         'Watch',
         'Plate',
         ]

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam  = cv2.VideoCapture(0)
count=0;
name = ''
names = ['x','y','z','v','w']
counter = 0
same = 0

while True:
    _,frame = cam.read()
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face = classifier.detectMultiScale(gray,1.3,5)
    #plt.imshow(frame)
    if face!=():
        for (x,y,w,h) in face:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            #cv2.putText(frame,'Human',(x,y))
            roi_gray = gray[y:y+h, x:x+w]
        cv2.imshow('Frame',frame)
        name='human'
        print(name)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if name=='human':
            count=count+1
            if count==5:
                print('confirm ',name)
                count=0
        else:
            count = 0

    else:
        cv2.imshow('Frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        img = np.asarray(frame)

        cv2.imwrite('temp_img.png',frame)
        img = cv2.imread('temp_img.png')
        
        img = cv2.resize(img,(100,100))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #img = cv2.equalizeHist(img)
        #img = cv2.bitwise_not(img)
        img =  img/255
        #img = img.reshape(1,784)
        img = np.reshape(img,(1,100,100,1))

        prediction = model.predict_classes(img)
        prob = model.predict_proba(img)
        max_prob = np.argmax(model.predict_proba(img))
        
        if model.predict_proba(img)[0][prediction][0]>0.99:
        #score = model.predict_proba(prediction)
            print("predicted class: ",objects[prediction[0]])#,'\tProb : ',prob)

            same = 0

            if counter == 5:
                counter = 0
                names = ['x','y','z','v','w']
            
            names[counter] = objects[prediction[0]]
            counter += 1

            for i in range(5):
                if names[0] == names[i]:
                    same+=1

            if same == 5:
                print("Confirm : ",objects[prediction[0]])
                same = 0
            print(same)
        

        

##        if name==prev:
##            
##            count=count+1
##            if count==5:
##                print('confirm',objects[prediction[0]])
##                count=0
##    


cam.release()    
cv2.destroyAllWindows()
    
