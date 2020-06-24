from scipy import *
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import profile
from cv2 import *
from scipy import ndimage

def constantDep(c,depMaxStr,depThreshold=0.08,depTransWidth=1/250,**kwargs):
    '''Constant deposition function'''
    return(0*c + depMaxStr)

def atanDep(c,depMaxStr,depThreshold=0.08,depTransWidth=1/250,**kwargs):
    '''arctan (soft switch) transition function'''
    return(depMaxStr/pi*(arctan((-c+depThreshold)/depTransWidth)+pi/2))

def atanDep2(c,depMaxStr,depThreshold=0.08,depTransWidth=1/250,**kwargs):
    '''arctan (soft switch) transition function with tanh'''
    return(depMaxStr/2*(np.tanh((-c+depThreshold)/depTransWidth)+1))

def linAtanDep(c,depMaxStr,depThreshold=0.08,depTransWidth=1/250,**kwargs):
    '''arctan (soft switch) transition function'''
    return(depMaxStr/pi*(c+.1*depThreshold)/(1.1*depThreshold)*(arctan((-c+depThreshold)/depTransWidth)+pi/2)**2)

def linAtanDep2(c,depMaxStr,depThreshold=0.08,depTransWidth=1/250,**kwargs):
    '''arctan (soft switch) transition function'''
    return(depMaxStr*(c+.2*depThreshold)/(depThreshold*2)*(np.tanh((-c+depThreshold)/depTransWidth)+1))
