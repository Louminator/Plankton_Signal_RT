from scipy import *
import numpy as np
from numpy.random import rand
import PlanktonSignaling.basicsSlice as PS
import PlanktonSignaling.Deposition as DP

gammas = np.array([1,2,3,4])*0.1
switch = np.array([1,2,3,4])*0.01
Deps = np.array([DP.atanDep2])
depp = ['atanDep2']
loops = 5
meshsize = 400
numb = 100**2
LL = 10
timesteps = 300

c0 = 0.012
def initial_conditions(x,y):
    return(0*x+c0)
M = 0
for dep in Deps:
	for B in range(0,len(gammas)):
		SumGradR = 0*linspace(0,1,timesteps+1)
		SumChem = 0*linspace(0,1,timesteps+1)
		f0 = dep(c0,gammas[B],switch[B],0.009)          
		p = 5*c0/f0
		for i in range(0,loops):
			Swimmers = PS.Plankton(dep,N = meshsize,depMaxStr=gammas[B],Const=3,L=LL,k=0.02,epsilon=1e-3,speed=1,
				lambda0=1,kappa=.2,beta=5,depThreshold=switch[B], depTransWidth=.009, dens=p, num = numb) 
			Swimmers.SetIC(initial_conditions)
		pos = np.empty((1,2))
		th = rand()*2*pi
		vel = [array([cos(th),sin(th)])]
		lenn = int(sqrt(numb))
		for l in range(0,lenn):
			for k in range(0,lenn):
				pos = np.append(pos,[array([mod(k*(Swimmers.L*1/lenn) + 0.01*(rand()-0.5) + (Swimmers.L*1/lenn),Swimmers.L), mod(l*(Swimmers.L*1/lenn) + 0.01*(rand()-0.5) + (Swimmers.L*1/lenn),Swimmers.L)])],axis=0)
				th = rand()*2*pi
				vel = np.append(vel,[array([cos(th),sin(th)])],axis=0)
		pos_store = list([pos[:,:]])
		pos_store = list([np.array(pos)])
		scalar_store = list([Swimmers.Meshed()])
		for plot in range(0,1):
			for k in range(0,timesteps):
				Swimmers.UpdateSlice(Swimmers.scalar,pos,vel)
				pos_store.append(np.array(pos))
				scalar_store.append(Swimmers.Meshed())
		#SUMIT  = []
		SUMC = []
		for i in range(len(scalar_store)):
			summp = 0
			summr = 0
		#	xs = pos_store[i][:,0]
		#	ys = pos_store[i][:,1]
		#	A2 = np.gradient(scalar_store[i],Swimmers.L/meshsize)
		#	for n in range(numb):
		#		xm = int(round(xs[n]))
		#		ym = int(round(ys[n]))
		#		summr = summr + np.sqrt(A2[0][xm,ym]**2 + A2[1][xm,ym]**2)
		#	SUMIT = np.append(SUMIT,summr)
			SUMC = np.append(SUMC,sum(scalar_store[i]))
		#SumGradR = SumGradR + SUMIT
		SumChem = SumChem + SUMC
	#np.save('{0}_Gamma{1}_Thres{2}_GradientRegN'.format(depp[M],gammas[B],switch[B]),SumGradR/loops)
		np.save('{0}_Gamma{1}_Thres{2}_ChemicalRegN'.format('A',gammas[B],switch[B]),SumChem/loops)
		print('Finish {0}'.format(B))
	M = M + 1
