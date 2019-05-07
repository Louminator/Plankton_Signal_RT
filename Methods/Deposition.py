from scipy import *
import numpy as np
import matplotlib.pyplot as plt
import profile

def constantDep(c,depMaxStr,**kwargs):
    '''Constant deposition function'''
    return(array(depMaxStr*ones(len(c))))

def atanDep(c,depMaxStr,depThreshold=0.08,depTransWidth=1/250,**kwargs):
    '''arctan (soft switch) transition function'''
    return(depMaxStr/pi*(arctan((-c+depThreshold)/depTransWidth)+pi/2))

def linAtanDep(c,depMaxStr,depThreshold=0.08,depTransWidth=1/250,**kwargs):
    '''arctan (soft switch) transition function'''
    return(depMaxStr/pi*(c+0.1*depThreshold)/1.1/depThreshold*(arctan((-c+depThreshold)/depTransWidth)+pi/2))

def expDep(c, depMaxStr, **kwargs):
    '''linear increase and hold constant at max'''
    return(depMaxStr / (1+exp(-3*c)) - depMaxStr/2.1)

def twoAtanDep(c, depMaxStr, depThreshold = 0.04, depThreshold2 = 0.8, depTransWidth=1/250, **kwargs):
    ''' combination of two Heaviside functions '''
    return(depMaxStr/pi*(arctan((-c+depThreshold2)/depTransWidth)+pi/2) * (depMaxStr/10 + depMaxStr/pi*(arctan((c-depThreshold)/depTransWidth)+pi/2) *depMaxStr/pi*(arctan((-c+depThreshold2)/depTransWidth)+pi/2)))
