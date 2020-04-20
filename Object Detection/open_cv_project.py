import cv2
import numpy as np
from sklearn import metrics,svm,tree,ensemble
from sklearn.neural_network import MLPClassifier
import time
MIN_MATCH_COUNT=30
objects = ['Book 1','Book 2','Telephone','Socket','Spoon','Fork']

#detector=cv2.xfeatures2d.SIFT_create()
detector = cv2.ORB_create()

FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})

trainImg=cv2.imread('./Book/u(1)/img(2).jpg',0)
trainKP,trainDesc=detector.detectAndCompute(trainImg,None)

xtrain = np.zeros((1,150000))
ytrain = np.zeros((96,1))
xtest = np.zeros((1,150000))
ytest = np.zeros((24,1))
count = 0
for i in range(1,7):
    for j in range(1,17):
        #print(count,i)
        img = cv2.imread('Book/u(%d)/img(%d).jpg'%(i,j),0)
        #print(img,img.shape)
        #train_img = np.resize(img,(400,400))
        KP,Desc=detector.detectAndCompute(img,None)
        Desc = np.resize(Desc,(500,300))
        data = np.reshape(Desc,(1,-1))
        xtrain = np.concatenate((xtrain,data),axis=0)
        ytrain[count] = i
        count+=1
        
count1 = 0
for i in range(1,7):
    for j in range(17,21):
        #print(count1,i)
        img = cv2.imread('Book/u(%d)/img(%d).jpg'%(i,j),0)
        #test_img = np.resize(img,(400,400))
        KP,Desc=detector.detectAndCompute(img,None)
        Desc = np.resize(Desc,(500,300))
        data = np.reshape(Desc,(1,-1))
        xtest = np.concatenate((xtest,data),axis=0)
        ytest[count1] = i
        count1+=1

xtrain = xtrain[1:,:]
xtest = xtest[1:,:]


###########################      NEURAL NETWORKS      #############################################


#neural_model = MLPClassifier(hidden_layer_sizes=(100,100,100),n_iter_no_change = 30)
#svm_model = svm.SVC()
RF_model = ensemble.RandomForestClassifier()
#DT_model = tree.DecisionTreeClassifier()
train_neural = RF_model.fit(xtrain,ytrain)
predict = train_neural.predict(xtest)
score = metrics.accuracy_score(ytest, predict)
conf_matrix = metrics.confusion_matrix(ytest,predict)

print(conf_matrix,'\n\nACCURACY :',score*100)

##cam = cv2.VideoCapture(0)
##
##while True:
##    ret,detect_img = cam.read()
##    gray = cv2.cvtColor(detect_img,cv2.COLOR_BGR2GRAY)
##    kp,features = detector.detectAndCompute(gray,None)
##    features = np.resize(features,(500,300))
##    cv2.imshow('fd',gray)
##    test_data = np.reshape(features,(1,-1))
##    prediction =  train_neural.predict(test_data)
##    print(objects[int(prediction)-1])
##    time.sleep(0.3)
##    
##
##    if cv2.waitKey(10) == ord('q'):
##        cam.release()
##        break
##
##
##cv2.destroyAllWindows()


cam=cv2.VideoCapture(0)
while True:
    ret, QueryImgBGR=cam.read()
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
    matches=flann.knnMatch(queryDesc,trainDesc,k=2)

    goodMatch=[]
    for m,n in matches:                      ### Where m is test matches and n is train image matches
        if(m.distance<0.75*n.distance):
            goodMatch.append(m)
    if(len(goodMatch)>MIN_MATCH_COUNT):
        tp=[]
        qp=[]
        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp,qp=np.float32((tp,qp))
        H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
        h,w=trainImg.shape
        trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        queryBorder=cv2.perspectiveTransform(trainBorder,H)
        cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)
    else:
        print ("Not Enough match found- %d/%d"%(len(goodMatch),MIN_MATCH_COUNT))
    cv2.imshow('result',QueryImgBGR)
    if cv2.waitKey(10)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
