# +
from scipy import *
import numpy as np
import profile

def constantDep(c,depMaxStr,depThreshold=0.08,depTransWidth=1/250,**kwargs):
    '''Constant deposition function'''
    return(depMaxStr+0*c)

def atanDep(c,depMaxStr,depThreshold=0.08,depTransWidth=1/250,**kwargs):
    '''arctan (soft switch) transition function with tanh'''
    return(depMaxStr/2*(np.tanh((-c+depThreshold)/depTransWidth)+1))

def linAtanDep(c,depMaxStr,depThreshold=0.08,depTransWidth=1/250,**kwargs):
    '''arctan (soft switch) transition function'''
    return(depMaxStr*(c+.2*depThreshold)/(depThreshold*2)*(np.tanh((-c+depThreshold)/depTransWidth)+1))
