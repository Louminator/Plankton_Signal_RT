from scipy import *
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib.animation
import profile

import scipy.sparse as sp
from scipy.optimize import broyden1
from scipy.interpolate import RectBivariateSpline,griddata
from scipy.sparse.linalg import spsolve
from numpy.random import rand, uniform
from numpy.linalg import inv
from scipy.linalg import toeplitz


# This module constructs all of the plankton and chemical matricies and #
# conducts the updating schemes for the PDEs that we are attempting to solve# 
        
class Background_Field(object):
    
    def __init__(self,N=217,d1=.1,d2=.1,left=0,right=3, dt=0.02):
        # initialize all values
        
        self.d1 = d1 # Diffusion rate for chemical 
        self.d2 = d2 # Decay rate for chemical
        self.N = N # Number of nodes for the spatial mesh
        self.left = left # Left endpoint of the spatial mesh
        self.right = right # Right endpoint of the spatial mesh
        self.dx = (right-left)/N # The spatial mesh size
        self.dt = dt # The temporal mesh size
        self.CC = 2*pi/(right - left) # Factor for Chebyshev matricies

        self.chebP() 
        self.BuildMatricies()
        
    def chebP(self):
        
        #construct the Chebyshev differentiation matricies with respect to the length of the interval
        
        h = (2*pi)/(self.N + 1) 
        i = array( range(1,(self.N + 1)) )
        column   = hstack(( [0.], .5*(-1)**i/tan(i*h/2.) ))
        row      = hstack(( [0.], column[self.N:0:-1]       ))
        D = toeplitz(column,row)

        x = zeros(self.N + 1)
        for i in range(0,self.N + 1):
            x[i]= self.left + i*h
        
        self.D = D*self.CC # The first Chebyshev differentiation matrix 
        self.xm = x/self.CC # The spatial discretization array
        self.D2 = D.dot(D) # The second Chebyshev differentiation matrix
     
    def BuildMatricies(self):
        
        # Building matricies used to conduct the Updating schemes for the PDE #
        
        # For Chemical
        self.I = eye(self.N + 1) 
        self.P = self.I + (self.dt/2)*(self.d1*self.D2 - self.d2*self.I) 
        self.M = inv(self.I - (self.dt/2)*(self.d1*self.D2 - self.d2*self.I))
    
        # Just For First Order System Plankton
        self.I2 = eye(2*(self.N+1))
        self.D1 = np.vstack((np.hstack((self.D, self.D*0)), np.hstack((self.D*0,self.D))))
        self.A = np.diag(np.append(-1*np.ones(self.N+1), np.ones(self.N+1)))
        self.AD1 = self.A.dot(self.D1)
    
    def initial_conditionsFO(self,const=0.012, steps=3000):
        
        #First Order intial conditions
        #Making initial matricies and setting up the totals using a steady state constant solution
        
        c = zeros((steps,self.N+1)) # chemical
        p = zeros((steps,self.N+1)) # right moving plankton
        q = zeros((steps,self.N+1)) # left moving plankton 
        CT = p[:,0]*0 # total chemical
        PT = p[:,0]*0 # total right moving plankton
        QT = p[:,0]*0 # total left moving plankton

        # construct steady state from PDE 
        c[0,:] = 0*np.linspace(0,10,self.N + 1) + const
        a = self.d2*const/(self.depFcn(const,self.depMaxStr,self.depThreshold,self.depTransWidth,*self.args,**self.kwargs))
        
        p[0,:] = a/2
        q[0,:] = a/2
        
        CT[0], PT[0], QT[0] = self.totalsF(c[0,:],p[0,:], q[0,:]) 
        
        return(c,p,q,CT,PT,QT)
    
    def initial_conditionsSO(self,const=0.012, steps=3000):
        
        #Second Order Initial Conditions
        #Making initial matricies and setting up the totals using a steady state constant solution
   
        
        c = zeros((steps,self.N+1)) # chemical 
        p = zeros((steps,self.N+1)) # plankton
        CT = p[:,0]*0 # total chemical
        PT = p[:,0]*0 # total plankton 

        c[0,:] = 0*np.linspace(0,10,self.N + 1) + const
        a = self.d2*c[0,:]/(self.depFcn(c[0,:],self.depMaxStr,self.depThreshold,self.depTransWidth,*self.args,**self.kwargs))
        p[0,:] = a
        
        CT[0], PT[0] = self.totals(c[0,:],p[0,:]) 
        
        return(c,p, CT, PT)
       
class PlankChem(Background_Field):
    
    def __init__(self,depFcn,depMaxStr=0.1,depThreshold=0.1, depTransWidth=0.1, delta = 1e-2,  *args,**kwargs):
        #initialize certain values 
        self.depMaxStr = depMaxStr #Deposition maximum strength
        self.depThreshold = depThreshold #Deposition for when the switch should 'turn off'
        self.depTransWidth = depTransWidth #Deposition 'smoothing' for the switch
        self.depFcn = depFcn #The Deposition function
        self.delta = delta #argument used to smooth out the sgn function c/sqrt(c^2 + delta^2) =~= sgn(c)
        self.args = args
        self.kwargs = kwargs
        
        super(PlankChem,self).__init__(*args,**kwargs)
        
    def firstStepFO(self,ck,pk,qk):
        
        # First Order Method
        # Perturb the constant solutions
        
        cn = ck + 0.00016*cos(2*self.CC*self.xm)
        pn = pk - 0.00014*cos(2*self.CC*self.xm)
        qn = qk + 0.00005*cos(2*self.CC*self.xm)
        
        totalc, totalp, totalq = self.totalsF(cn,pn,qn)
        
        return(cn,pn,qn,totalc,totalp,totalq)
        
    def firstStepSO(self,ck,pk):
        # Second Order Method 
        # Perturb the constant solutions 

        cn = ck + 0.00016*cos(2*self.CC*self.xm)
        pn = pk - 0.00009*cos(2*self.CC*self.xm)
        
        totalc, totalp = self.totals(cn,pn)
        
        return(cn,pn,totalc,totalp)
    
    def FirstOrder(self,ck,pk,qk):
        #this is the updating scheme for the first order equation#
        #We utilize a Crank-Nicolson scheme for both the chemical and plankton, but using a psuedospectral scheme#
        #We compute the change in the chemical first and then use this to perform the next evolution of the plankton#
        
        
        #timestep for chemical
        F = np.multiply(pk+qk,self.depFcn(ck,self.depMaxStr,self.depThreshold,self.depTransWidth,*self.args,**self.kwargs))
        cp = self.dt*F
        B = self.P.dot(ck) + cp
        cn = self.M.dot(B)
        
        #timestep for plankton
        PSI = np.append(pk,qk)
        
        if (self.delta == 0): #this is to prevent any division by zero
            SGN = np.sign(self.D.dot(ck))  
            SGN1 = np.sign(self.D.dot(ck1))
        else:
            SGN = np.divide(self.D.dot(ck),sqrt(np.multiply(self.D.dot(ck),self.D.dot(ck))+self.delta**2))
            SGN1 = np.divide(self.D.dot(cn),sqrt(np.multiply(self.D.dot(cn),self.D.dot(cn))+self.delta**2))

        #Construt the Sign Matricies here by using vstack
        C1 = np.diag(np.ones(len(ck)) - SGN)
        C2 = np.diag(np.ones(len(ck)) + SGN)
        C3 = -1*C1
        C4 = -1*C2

        Hn = np.vstack((np.hstack((C3, C2)),np.hstack((C1,C4))))

        C11 = np.diag(np.ones(len(ck)) - SGN1)
        C12 = np.diag(np.ones(len(ck)) + SGN1)
        C13 = -1*C1
        C14 = -1*C2

        Hn1 = np.vstack((np.hstack((C13, C12)),np.hstack((C11,C14))))
        
        
        #Finish off the method by inverting the proper matrix
        K = (self.I2 + (self.dt/2)*(self.AD1 + Hn))
        K1 = K.dot(PSI)
        PSIN = inv(self.I2 - (self.dt/2)*(self.AD1 + Hn1)).dot(K1)

        pn = PSIN[:len(PSIN)//2]
        qn = PSIN[len(PSIN)//2:]
        
        #Compute the total amount of plankton and chemical currently in the system
        totalc, totalp, totalq = self.totalsF(cn,pn,qn)
        
        return(cn,pn,qn,totalc,totalp,totalq)
    
    def SecondOrder(self,ck,pk,pk1):
        #this is the updating scheme for the second order equation#
        #We utilize a Crank-Nicolson scheme for both the chemical and plankton, but using a psuedospectral scheme#
        #We compute the change in the chemical first and then use this to perform the next evolution of the plankton#    
        
        
        #calculate timestep for c
        F = np.multiply(pk,self.depFcn(ck,self.depMaxStr,self.depThreshold,self.depTransWidth,*self.args,**self.kwargs))
        cp = self.dt*F
        B = self.P.dot(ck) + cp
        cn = self.M.dot(B)

        #calculate next timestep for p
        if (self.delta==0): #prevent possible division by 0
            first = np.multiply(np.sign(D.dot(ck)),pk)
        else:  
            first = np.multiply(np.divide(self.D.dot(ck),sqrt(np.multiply(self.D.dot(ck),self.D.dot(ck))+self.delta**2)),pk)
            
        mp = self.D.dot(first)
        fp = (self.dt**2)*(self.D2.dot(pk) - mp)
        pn = (1/(1 + self.dt/2))*(fp + 2*pk + (self.dt/2 - 1)*pk1)
           
        #compute the sum of total chemical and plankton
        C,P = self.totals(cn,pn)
        
        return(cn,pn,C,P)   
    
    def totals(self,ck,pk):
        #Computes the total number of plankton and chemical in the second order system using the trapezoidal rule#
        
        totalp = 0
        totalc = 0
        for j in range(1,self.N+1):
            Q = (pk[j]+pk[j-1])/2
            Z = (ck[j]+ck[j-1])/2
            totalp = Q*self.dx + totalp
            totalc = Z*self.dx + totalc
        return(totalc, totalp)
   
    def totalsF(self,ck,pk,qk):
        #Computes the total number of plankton and chemical in the first order system using the trapezoidal rule#

        totalp = 0
        totalq = 0
        totalc = 0
        for j in range(1,self.N+1):
            Q = (pk[j]+pk[j-1])/2
            H = (qk[j]+qk[j-1])/2
            Z = (ck[j]+ck[j-1])/2
            totalp = Q*self.dx + totalp
            totalq = H*self.dx + totalq
            totalc = Z*self.dx + totalc
        return(totalc, totalp, totalq)
        