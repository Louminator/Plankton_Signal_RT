# coding: utf-8

from scipy import *
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import profile
from cv2 import *
from scipy import ndimage

def constantDep(c,depMaxStr,**kwargs):
    '''Constant deposition function'''
    return(array(depMaxStr*ones(len(c))))

def atanDep(c,depMaxStr,depThreshold=0.08,depTransWidth=1/250,**kwargs):
    '''arctan (soft switch) transition function'''
    return(depMaxStr/pi*(arctan((-c+depThreshold)/depTransWidth)+pi/2))

def linAtanDep(c,depMaxStr,depThreshold=0.08,depTransWidth=1/250,**kwargs):
    '''arctan (soft switch) transition function'''
    return(depMaxStr/pi*(c+0.1*depThreshold)/1.1/depThreshold*(arctan((-c+depThreshold)/depTransWidth)+pi/2))


def blobdetect(thresh, maxValue, swimmers):
    im2 = np.array(swimmers*255/0.08, dtype = np.uint8)
    threshed = cv2.adaptiveThreshold(im2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    plt.imshow(im2)

    th, dst = cv2.threshold(im2, thresh, maxValue, cv2.THRESH_BINARY)
    plt.imshow(dst)
    im = cv2.bitwise_not(dst)

    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 0
    params.maxThreshold = 300
    params.filterByConvexity = False
    params.filterByArea = False

    detector = cv2.SimpleBlobDetector_create(params)

    #Detect the blobs

    keypoints = detector.detect(im)

    im=cv2.bitwise_not(im)

    #plot the blob contours

    #im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #plt.imshow(im_with_keypoints)

    #outputs how many blobs there are 

    return(len(keypoints));

def blobdetect2(thresh, maxValue, swimmers):
    im2 = np.array(swimmers*255/0.08, dtype = np.uint8)
    im3 = np.tile(im2,(2,2))
    threshed = cv2.adaptiveThreshold(im3, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    #plt.imshow(im2)

    th, dst = cv2.threshold(im3, thresh, maxValue, cv2.THRESH_BINARY)
    #plt.imshow(dst)
    im = cv2.bitwise_not(dst)

    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 0
    params.maxThreshold = 300
    params.filterByConvexity = False
    params.filterByArea = False

    detector = cv2.SimpleBlobDetector_create(params)

    #Detect the blobs

    keypoints = detector.detect(im)

    im=cv2.bitwise_not(im)

    #plot the blob contours

    #im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #plt.imshow(im_with_keypoints)

    #outputs how many blobs there are 

    return(len(keypoints));

def blobdetect3(thresh, maxValue, swimmers):
    im2 = np.array(swimmers*255/0.08, dtype = np.uint8)
    im3 = np.tile(im2,(3,3))
    threshed = cv2.adaptiveThreshold(im3, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    #plt.imshow(im2)

    th, dst = cv2.threshold(im3, thresh, maxValue, cv2.THRESH_BINARY)
    #plt.imshow(dst)
    im = cv2.bitwise_not(dst)

    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 0
    params.maxThreshold = 300
    params.filterByConvexity = False
    params.filterByArea = False

    detector = cv2.SimpleBlobDetector_create(params)

    #Detect the blobs

    keypoints = detector.detect(im)

    im=cv2.bitwise_not(im)

    #plot the blob contours

    #im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #plt.imshow(im_with_keypoints)

    #outputs how many blobs there are 

    return(len(keypoints));

def countblob(a,b,c):
    A = np.array([[1,2,4],[4,3,9],[9,12,16]])
    B = np.array([[a],[b],[c]])
    x = np.linalg.solve(A,B)
    return(x.sum());
