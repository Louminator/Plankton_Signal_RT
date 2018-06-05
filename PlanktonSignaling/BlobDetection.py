# coding: utf-8

from scipy import *
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import profile
from cv2 import *
from scipy import ndimage

def countingAll(thresh, maxValue, swimmers):
    fir = countblobNew(0,thresh,maxValue,swimmers)
    sec = countblobNew(255,thresh,maxValue,swimmers)
    return(fir + sec);

def countblobNew(M,thresh, maxValue, swimmers):
    a = blobdetectNew(M,1,thresh, maxValue, swimmers)
    b = blobdetectNew(M,2,thresh, maxValue, swimmers)
    c = blobdetectNew(M,3,thresh, maxValue, swimmers)
    A = np.array([[1,2,4],[4,6,9],[9,12,16]])
    B = np.array([[a],[b],[c]])
    x = np.linalg.solve(A,B)
    return(x.sum());

def blobdetectNew(M, N, thresh, maxValue, swimmers):
    im2 = np.array(swimmers*255/0.08, dtype = np.uint8)
    im3 = np.tile(im2,(N,N))
    #threshed = cv2.adaptiveThreshold(im3, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)

    th, dst = cv2.threshold(im3, thresh, maxValue, cv2.THRESH_BINARY)
    #plt.imshow(dst)
    im = cv2.bitwise_not(dst)

    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 0
    params.maxThreshold = 300
    params.filterByConvexity = False
    params.filterByArea = False
    params.blobColor = M

    detector = cv2.SimpleBlobDetector_create(params)

    #Detect the blobs

    keypoints = detector.detect(im)

    #im=cv2.bitwise_not(im)

    #plot the blob contours

    #im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #plt.imshow(im_with_keypoints)

    #outputs how many blobs there are 

    return(len(keypoints));