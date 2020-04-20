import cv2
import numpy as np

img = cv2.imread('Book/u(1)/img(15).jpg',cv2.IMREAD_GRAYSCALE)
MIN_MATCH_COUNT = 30

objects = ['Book 1','Book 2','Telephone']

sift = cv2.xfeatures2d.SIFT_create()
#surf = cv2.xfeatures2d_SURF.create()
orb = cv2.ORB_create()

FLANN_INDEX_KDTREE = 0
flann_param = dict(algorithm = FLANN_INDEX_KDTREE,tree=5)
flann = cv2.FlannBasedMatcher(flann_param,{})

keypoints, descriptors = sift.detectAndCompute(img,None)


cam = cv2.VideoCapture(0)
while True:
    _, frame = cam.read()
    test_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    kp, desc = sift.detectAndCompute(test_img,None)

    matches = flann.knnMatch(desc,descriptors,k=2)

    goodMatch = []
    for m,n in matches:
        if(m.distance<0.75*n.distance):
            goodMatch.append(m)
    if(len(goodMatch)>MIN_MATCH_COUNT):
        tp=[]
        qp=[]

        for m in goodMatch:
            tp.append(keypoints[m.trainIdx].pt)
            qp.append(kp[m.queryIdx].pt)
        tp,qp = np.float32((tp,qp))
        H, status = cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
        h,w = img.shape
        trainBorder = np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        queryBorder = cv2.perspectiveTransform(trainBorder,H)
        cv2.polylines(frame,[np.int32(queryBorder)],True,(0,255,0),5)
    else:
        print('Not Enough MAtches = %d/%d'%(len(goodMatch),MIN_MATCH_COUNT))
    cv2.imshow('result',frame)
    if cv2.waitKey(10)==ord('q'):
        break


#img = cv2.drawKeypoints(img,keypoints,None)



##cv2.imshow('Image',img)
cam.release()
cv2.destroyAllWindows()
