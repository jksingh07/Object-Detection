import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimage
import cv2
from sklearn.externals import joblib
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D,Dropout,Conv2D,Flatten
from keras.optimizers import Adam


train_data = []
train_label = []
valid_data = []
valid_label = []
test_data = []
test_label = []

##for i in range(4):
##    train_data.append([])


def grayscale(img):
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  return gray
  
def equalize(gray):   
  img = cv2.equalizeHist(gray) # it will only take gray scale images and distribute the intensity over whole and increases the brightness of the images
  return img

def preprocessing(img):
  img =  grayscale(img)
  img = equalize(img)
  img = img/255  # this  is called normalisation to make data b/w 0-1
  return img

for i in range(10):
    for j in range(40):
        path = 'C:/Users/Lenovo/Desktop/Project Dataset/u(%d)/img(%d).jpg'%(i+1,j+1)
        img = mimage.imread(path)
        #img = cv2.resize(img,(32,32))
        #img = preprocessing(img)
        train_data.append(img)
        train_label.append(i)

for i in range(10):
    for j in range(40,50):
        path = 'C:/Users/Lenovo/Desktop/Project Dataset/u(%d)/img(%d).jpg'%(i+1,j+1)
        img = mimage.imread(path)
        #img = cv2.resize(img,(32,32))
        #img = preprocessing(img)
        valid_data.append(img)
        valid_label.append(i)

for i in range(10):
    for j in range(50,60):
        path = 'C:/Users/Lenovo/Desktop/Project Dataset/u(%d)/img(%d).jpg'%(i+1,j+1)
        img = mimage.imread(path)
        #img = cv2.resize(img,(32,32))
        #img = preprocessing(img)
        test_data.append(img)
        test_label.append(i)


train_data = np.asarray(train_data)
train_label = np.asarray(train_label)

valid_data = np.asarray(valid_data)
valid_label = np.asarray(valid_label)

test_data = np.asarray(test_data)
test_label = np.asarray(test_label)

## For saving the pickle file 

##joblib.dump(train_data,'C:\\Users\\Lenovo\\Desktop\\Project Dataset\\train.pkl')
##joblib.dump(train_label,'C:\\Users\\Lenovo\\Desktop\\Project Dataset\\train_label.pkl')
##
##joblib.dump(test_data,'C:\\Users\\Lenovo\\Desktop\\Project Dataset\\test.pkl')
##joblib.dump(test_label ,'C:\\Users\\Lenovo\\Desktop\\Project Dataset\\test_label.pkl')
##
##joblib.dump(valid_data,'C:\\Users\\Lenovo\\Desktop\\Project Dataset\\valid.pkl')
##joblib.dump(valid_label,'C:\\Users\\Lenovo\\Desktop\\Project Dataset\\valid_label.pkl')
##
p=joblib.load('C:\\Users\\Lenovo\\Desktop\\Project Dataset\\train.pkl')
print(p.shape)


X_train = np.array(list(map(preprocessing,train_data)))
X_train = X_train.reshape(400, 32, 32,1)
y_train = train_label

X_valid = np.asarray(list(map(preprocessing,valid_data)))
X_valid = X_valid.reshape(100, 32, 32,1)
y_valid = valid_label

X_test = np.asarray(list(map(preprocessing,test_data)))
X_test = X_test.reshape(100, 32, 32,1)
y_test = test_label

# For Making More dataset
datagen = ImageDataGenerator(width_shift_range=0.1,
                  height_shift_range=0.1,
                  zoom_range=0.2,
                  shear_range=0.1,
                  rotation_range=30)

datagen.fit(X_train)

num_classes = 10

# One Hot Encoding for labels
y_train = to_categorical(y_train,10)
y_valid = to_categorical(y_valid,10)
y_test = to_categorical(y_test,10)


# CONVOLUTIONAL NEURAL NETWORK MODEL LE-NET

def modified_net():
  model = Sequential()
  model.add(Conv2D(60,(5,5),input_shape=(32,32,1),activation='relu'))
  model.add(Conv2D(60,(5,5),activation='relu'))
  model.add(MaxPooling2D(pool_size = (2,2)))
  
  model.add(Conv2D(30,(3,3),activation='relu'))
  model.add(Conv2D(30,(3,3),activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.5))
  
  model.add(Flatten())
  model.add(Dense(units = 500,activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes,activation ='softmax'))
  # COMPILE MODEL
  model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
  
  return model

model = modified_net()
print(model.summary())

joblib.dump(model,'C:\\Users\\Lenovo\\Desktop\\Project Dataset\\model.pkl')

history = model.fit_generator(datagen.flow(X_train,y_train,batch_size=50),steps_per_epoch = 300,epochs=10,validation_data=(X_valid,y_valid))

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.xlabel('epochs')
plt.title('Loss')

score = model.evaluate(X_test,y_test)
print('Test Score :',score[0],'Test Accuracy :',score[1])

plt.show()



