
# coding: utf-8

from scipy import *
import scipy.sparse as sp
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import RectBivariateSpline
from scipy.sparse.linalg import spsolve
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib.animation
from scipy.interpolate import RectBivariateSpline

class Background_Field(object):
    "A class that creates the background concentration field and evolves"
    
    # class builder initiation 
    def __init__(self,N=30,kappa=1.0e-4,beta=1.0e0,k=0.1,*args,**kwargs):
        self.N = N # The number of mesh points
        self.kappa = kappa
        self.beta  = beta
        self.k     = k # k is delta t
        self.x = r_[0:1:1j*self.N]# setup the spatial mesh. It is a long row vector

        # Create some local coordinates for the square domain.
        self.y = 1*self.x
        self.xm,self.ym = np.meshgrid(self.x,self.y)
        
        self.scalar = self.x
        for counter in range(0,self.N-1):
            self.scalar = np.append(self.scalar,self.x)
        
        self.h = self.x[1]-self.x[0] # spacial mesh size
        self.SetAlpha()
        self.BuildMatrixA1()
        self.BuildMatrixA2()
        self.BuildMatrices() # build M1 and M2
    
    # Forward one time step. Just diffuse
    def Update(self, vectors): # Remark: we feed in all info, but only those flagged
        # will be emitting substances to change the background field
        
        self.scalar = spsolve(self.M1, self.M2.dot(vectors))

        return self.scalar
    
    # Set up initial condition
    def SetIC(self,f):
        ic = f(self.xm,self.ym)
        ic = ic.reshape((self.N**2,))
        self.IC = ic
        self.scalar = ic
        
    def Meshed(self):
        return(self.scalar.reshape((self.N,self.N)))
        
    # Compute alpha
    def SetAlpha(self):
        self.alpha = self.kappa*self.k/self.h/self.h
        self.BuildMatrixA1()
        self.BuildMatrixA2()
        self.BuildMatrices()
        
    def SetBeta(self,beta):
        self.beta = beta
        self.BuildMatrixA1()
        self.BuildMatrixA2()
        self.BuildMatrices()
        
    # Build the N x N matrix A1 for 1-Dimensional Crank-Nicoleson Method
    def BuildMatrixA1(self):
        diag = ones(self.N)*(1+4*self.alpha/2+self.k*self.beta/2)
        data = np.array([-ones(self.N)*self.alpha/2,-ones(self.N)*self.alpha/2,
                         diag, -ones(self.N)*self.alpha/2,-ones(self.N)*self.alpha/2]) #off-diag and corners are -alpha
        self.A1 = sp.spdiags(data,[1-self.N,-1,0,1,self.N-1],self.N,self.N)
        
    def BuildMatrixA2(self):
        diag = ones(self.N)*(1-4*self.alpha/2-self.k*self.beta/2)
        data = np.array([ones(self.N)*self.alpha/2, ones(self.N)*self.alpha/2,
                         diag, ones(self.N)*self.alpha/2, ones(self.N)*self.alpha/2]) #off-diag and corners are alpha
        self.A2 = sp.spdiags(data,[1-self.N,-1,0,1,self.N-1],self.N,self.N)
    
    # Build the big matrices M1 M2 using I, A1 and A2
    def BuildMatrices(self):
        ############ Build M1
        self.I = sp.identity(self.N) # Identity N x N Sparse Matrix
        self.E = sp.csr_matrix((self.N,self.N)) # Zero N x N Sparse Matrix
        Rows = {i: self.E for i in range(self.N)} # Empty rows of tile matrices
        Rows[0] = self.A1
        Rows[1] = -self.I*self.alpha/2
        Rows[self.N-1] = -self.I*self.alpha/2
        # Creating rows
        for i in range(self.N):
            for j in range(1,self.N):
                if j == i:
                    buildblock = self.A1
                elif j == (i-1 % self.N) or j == (i+1 % self.N) or (j==self.N-1 and i==0): # a cheap way to fix
                    buildblock = -self.I*self.alpha/2
                else:
                    buildblock = self.E
                Rows[i] = sp.hstack([Rows[i],buildblock]) # Stack matrices horizontally to create rows
                
        # Stack rows together vertically to get M1
        self.M1 = Rows[0]
        for i in range(1,self.N):
            self.M1 = sp.vstack([self.M1,Rows[i]])
        self.M1 = self.M1.tocsr()    
        ############ Build M2
        Rows = {i: self.E for i in range(self.N)} # Empty rows of tile matrices
        Rows[0] = self.A2
        Rows[1] = self.I*self.alpha/2
        Rows[self.N-1] = self.I*self.alpha/2
        # Creating rows
        for i in range(self.N):
            for j in range(1,self.N):
                if j == i:
                    buildblock = self.A2
                elif j == (i-1 % self.N) or j == (i+1 % self.N) or (j==self.N-1 and i==0): # a cheap way to fix
                    buildblock = self.I*self.alpha/2
                else:
                    buildblock = self.E
                Rows[i] = sp.hstack([Rows[i],buildblock]) # Stack matrices horizontally to create rows
                
        # Stack rows together vertically to get M2
        self.M2 = Rows[0]
        for i in range(1,self.N):
            self.M2 = sp.vstack([self.M2,Rows[i]])
        self.M2 = self.M2.tocsr()
        
    def CheckM():
        checkmatrix = self.M1.toarray()
        print(np.array2string(checkmatrix,precision=2))
        checkmatrix = self.M2.toarray()
        print(np.array2string(checkmatrix,precision=2))
        
    def CheckA():
        checkmatrix = self.A1.toarray()
        print(np.array2string(checkmatrix,precision=2))
        checkmatrix = self.A2.toarray()
        print(np.array2string(checkmatrix,precision=2))


class Plankton(Background_Field):
    
    def __init__(self,depFcn,lambda0=1e0,speed=0.1,depMaxStr=1.0e-10,depVar=1.0e-5,epsilon=1.0e-8,*args,**kwargs):

        self.lambda0 = lambda0
        self.speed = speed
        self.depMaxStr = depMaxStr #Deposition maximum strength
        self.depVar = depVar       #Deposition variable (Gaussian deposition)
        self.depFcn = depFcn
        self.args = args
        self.kwargs = kwargs
        
        self.epsilon = epsilon

        super(Plankton,self).__init__(*args,**kwargs)

        print('Exact deposition variance: {0:8.2e}, length scale: {1:8.2e}.  a2: {2:8.2e}.'.format(self.k*self.kappa,
                                                                                             sqrt(self.k*self.kappa),
                                                                                             self.depVar))
        
    def RT(self,pos,vel,c,grad_c):
        # Actually, I need to do this as tumble and run, TR.
        for j in range(0,len(pos)):
            alpha = 1/(self.epsilon + sqrt(grad_c[j].dot(grad_c[j])*vel[j].dot(vel[j])))
            if (rand() < self.k*self.lambda0*0.5*(1-alpha*grad_c[j].dot(vel[j]))):
                th = rand()*2*pi
                vel[j] = self.speed*array([cos(th),sin(th)])
        for j in range(0,len(pos)):
            pos[j] += self.k*vel[j]
            pos[j] = mod(pos[j],1)
            
    def RT2(self,pos,vel,c,grad_c):
        # Actually, I need to do this as tumble and run, TR.
        for j in range(0,len(pos)):
            alpha = 1/(self.epsilon + sqrt(dot(grad_c[j],grad_c[j])*dot(vel[j],vel[j])))
            if (rand() < self.k*self.lambda0*0.5*(1-alpha*dot(vel[j],grad_c[j]))):
                th = rand()*2*pi
                vel[j] = self.speed*array([cos(th),sin(th)])
        for j in range(0,len(pos)):
            pos[j] += self.k*vel[j]
            pos[j] = mod(pos[j],1)
        
    def Update(self,vectors,pos,vel):
        c      = self.scalarInterp(pos)
        grad_c = self.scalarGrad(pos)
        self.RT(pos,vel,c,grad_c)
        
        depStr = self.depFcn(c,self.depMaxStr,*self.args,**self.kwargs)
        f = zeros((self.N,self.N))
        for p,str in zip(pos,depStr):
            f = f + str*exp(-((self.xm-p[0])**2+(self.ym-p[1])**2)/4/self.depVar)/(4*pi*self.depVar)
            # Be cautious about periodic BC's.
            # We capture periodic source emissions.
            # Assumes a [0,1]x[0,1] domain.
            if (p[0]<8*sqrt(self.depVar)):
                f = f + str*exp(-((self.xm-p[0]-1)**2+(self.ym-p[1])**2)/4/self.depVar)/(4*pi*self.depVar)
            if (p[0]>1-8*sqrt(self.depVar)):
                f = f + str*exp(-((self.xm-p[0]+1)**2+(self.ym-p[1])**2)/4/self.depVar)/(4*pi*self.depVar)
            if (p[1]<8*sqrt(self.depVar)):
                f = f + str*exp(-((self.xm-p[0])**2+(self.ym-p[1]-1)**2)/4/self.depVar)/(4*pi*self.depVar)
            if (p[1]>1-8*sqrt(self.depVar)):
                f = f + str*exp(-((self.xm-p[0])**2+(self.ym-p[1]+1)**2)/4/self.depVar)/(4*pi*self.depVar)
        f = f.reshape((self.N*self.N,))
        self.scalar = spsolve(self.M1, self.M2.dot(vectors)+self.k*f)
        return(self.scalar)
    
    def Update2(self,vectors,pos,vel):
        c      = self.scalarInterp2(pos)
        grad_c = self.scalarGrad2(pos)
        self.RT2(pos,vel,c,grad_c)
        
        depStr = self.depFcn(c,self.depMaxStr,*self.args,**self.kwargs)
        f = zeros((self.N,self.N))
        for p,str in zip(pos,depStr):
            f = f + str*exp(-((self.xm-p[0])**2+(self.ym-p[1])**2)/4/self.depVar)/(4*pi*self.depVar)
            # Be cautious about periodic BC's.
            # We capture periodic source emissions.
            # Assumes a [0,1]x[0,1] domain.
            if ((p[0])**2<64*self.depVar):
                f = f + str*exp(-((self.xm-p[0]-1)**2+(self.ym-p[1])**2)/4/self.depVar)/(4*pi*self.depVar)
            if ((p[0]-1)**2>64*self.depVar):
                f = f + str*exp(-((self.xm-p[0]+1)**2+(self.ym-p[1])**2)/4/self.depVar)/(4*pi*self.depVar)
            if ((p[1])**2<64*self.depVar):
                f = f + str*exp(-((self.xm-p[0])**2+(self.ym-p[1]-1)**2)/4/self.depVar)/(4*pi*self.depVar)
            if ((p[1]-1)**2>64*self.depVar):
                f = f + str*exp(-((self.xm-p[0])**2+(self.ym-p[1]+1)**2)/4/self.depVar)/(4*pi*self.depVar)
        f = f.reshape((self.N*self.N,))
        self.scalar = spsolve(self.M1, self.M2.dot(vectors)+self.k*f)
        return(self.scalar)
    
    def scalarInterp(self,p):
        return(griddata((self.xm.reshape(self.N**2,),self.ym.reshape(self.N**2,)),self.scalar,p,method='cubic'))

    def scalarInterp2(self,pos):
        bspline = RectBivariateSpline(self.x,self.y,self.Meshed())
        return(bspline.ev(pos[:,0],pos[:,1]))
    
    def scalarInterp3(self,p):
        bspline = RectBivariateSpline(self.x,self.y,self.Meshed())
        p = array(p)
        what = bspline.ev(p[:,:,0],p[:,:,1])
        #what = what.reshape((4,len(p)/8))
        return(what)
    
    # Assumes a [0,1]x[0,1] domain.
    def scalarGrad(self,xp,dx=1.0e-4):
        dp = array(self.scalarInterp([mod(xp + array([dx,0]),1),mod(xp - array([dx,0]),1),mod(xp + array([0,dx]),1),
                                      mod(xp - array([0,dx]),1)]))
        diffs = array([dp[0]-dp[1],dp[2]-dp[3]])/2/dx
        diffs = diffs.T
        return(diffs)
    
    def scalarGrad2(self,xp,dx=1.0e-4):
        dp = self.scalarInterp3([mod(xp + array([dx,0]),1),mod(xp - array([dx,0]),1),mod(xp + array([0,dx]),1),
                                      mod(xp - array([0,dx]),1)])
        diffs = array([dp[0]-dp[1],dp[2]-dp[3]])/2/dx
        diffs = diffs.T
        
        return(diffs)





