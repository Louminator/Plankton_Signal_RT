# coding: utf-8

from scipy import *
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import profile
from cv2 import *
from scipy import ndimage

def Contour(Swimmers,pos,N,meshsize):
    f = zeros((meshsize,meshsize))
    Std = N/Swimmers.L**2
    
    for i in range(0,N):
        A = 0
        B = 0
        C = 0
        D = 0
        
        p = pos[i,:]
        f = f + (1/(4*pi*Std))*exp(-((Swimmers.xm-p[0])**2+(Swimmers.ym-p[1])**2)/4/Std)/(4*pi*Std)
        if ((p[0])**2<Std):
            f = f + (1/(4*pi*Std))*exp(-((Swimmers.xm-p[0]-Swimmers.L)**2+(Swimmers.ym-p[1])**2)/4/Std)/(4*pi*Std)
            A = 1
            
        if ((p[0]-Swimmers.L)**2>Std):
            f = f + (1/(4*pi*Std))*exp(-((Swimmers.xm-p[0]+Swimmers.L)**2+(Swimmers.ym-p[1])**2)/4/Std)/(4*pi*Std)
            B = 1
            
        if ((p[1])**2<Std):
            f = f + (1/(4*pi*Std))*exp(-((Swimmers.xm-p[0])**2+(Swimmers.ym-p[1]-Swimmers.L)**2)/4/Std)/(4*pi*Std)
            C = 1
            
        if ((p[1]-Swimmers.L)**2>Std):
            f = f + (1/(4*pi*Std))*exp(-((Swimmers.xm-p[0])**2+(Swimmers.ym-p[1]+Swimmers.L)**2)/4/Std)/(4*pi*Std)
            D = 1
            
        if (A == 1 and C == 1): #Plankton in Lower Left Corner
            f = f + (1/(4*pi*Std))*exp(-((Swimmers.xm-p[0]-Swimmers.L)**2+(Swimmers.ym-p[1]-Swimmers.L)**2)/4/Std)
        if (A == 1 and D == 1): #Plankton in Upper Left Corner
            f = f + (1/(4*pi*Std))*exp(-((Swimmers.xm-p[0]-Swimmers.L)**2+(Swimmers.ym-p[1]+Swimmers.L)**2)/4/Std)
        if (B == 1 and C == 1): #Plankton in Upper Right Corner
            f = f + (1/(4*pi*Std))*exp(-((Swimmers.xm-p[0]+Swimmers.L)**2+(Swimmers.ym-p[1]-Swimmers.L)**2)/4/Std)
        if (B == 1 and D == 1): #Plankton in Lower Right Corner
            f = f + (1/(4*pi*Std))*exp(-((Swimmers.xm-p[0]+Swimmers.L)**2+(Swimmers.ym-p[1]+Swimmers.L)**2)/4/Std)
                                       
        AA = 0*f
        for j in range(0,meshsize):
            for k in range(0,meshsize):
                if (f[j,k]>2):
                    AA[j,k] = 1
                                       
    return(AA)

def countingAll(thresh, maxValue, swimmers):
    fir = countblobNew(0,thresh,maxValue,swimmers)
    sec = countblobNew(255,thresh,maxValue,swimmers)
    return(fir + sec);
    #return(fir);

def countblobNew(M,thresh, maxValue, swimmers):
    a = blobdetectNew(M,1,thresh, maxValue, swimmers)
    b = blobdetectNew(M,2,thresh, maxValue, swimmers)
    c = blobdetectNew(M,3,thresh, maxValue, swimmers)
    A = np.array([[1,2,4],[4,6,9],[9,12,16]])
    B = np.array([[a],[b],[c]])
    x = np.linalg.solve(A,B)
    return(x.sum());

def blobdetectNew(M, N, thresh, maxValue, swimmers):
    im2 = np.array(swimmers*255, dtype = np.uint8)
    im3 = np.tile(im2,(N,N))

    th, dst = cv2.threshold(im3, thresh, maxValue, cv2.THRESH_BINARY)
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

    return(len(keypoints));