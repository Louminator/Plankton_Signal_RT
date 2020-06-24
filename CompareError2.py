from scipy import *
import numpy as np
from numpy.random import rand
import PlanktonSignaling.basicsFixSlice as PS
import PlanktonSignaling.Deposition as DP

meshsize = 400
timesteps = 600

def initial_conditions(x,y):
	return(0.012 + 1e-6*sin(8*pi*x/5))
	#return(0.012 + 0*x)

numb = 160000
LL = 10
d1 = .2
d2 = 5
c0 = 0.012
f0 = DP.constantDep(c0,0.01,0.02,0.003)
p = d2*c0/f0
K = 4
Swimmers = PS.Plankton(DP.constantDep,N = meshsize,depMaxStr=.01,Const=3,L=LL,k=0.02,epsilon=1e-3,speed=1,
		lambda0=1,kappa=.2,beta=5,depThreshold=0.02, depTransWidth=0.003, dens=p, num = numb)

Swimmers.SetIC(initial_conditions)

pos = np.empty([1,2])

lenn= int(sqrt(numb))
th = rand()*2*pi
vel = [array([cos(th),sin(th)])]

for l in range(0,lenn):
	for k in range(0,lenn):
		pos = np.append(pos,[array([mod(k*(Swimmers.L*1/lenn) + 0.01*(rand()-0.5) + 0.5*(Swimmers.L*1/lenn),Swimmers.L), mod(l*(Swimmers.L*1/lenn) + 0.01*(rand()-0.5) + 0.5*(Swimmers.L*1/lenn),Swimmers.L)])],axis=0)
		th  = rand()*2*pi
		vel = np.append(vel,[array([cos(th),sin(th)])],axis=0)

pos2 = np.empty_like(pos)
pos2[:,:] = pos
pos_store = list([pos[:,:]])
pos_store = list([np.array(pos)])

pos_storeinit = list([pos[:,:]])
pos_storeinit = list([np.array(pos)])
vel2 = np.empty_like(vel)
vel2[:,:] = vel

#Seeding with most unstable wave number

scalar_store = list([Swimmers.Meshed()]) 
for m in range(0,K):
	Swimmers.Update(Swimmers.scalar,pos,vel)
	pos_store.append(np.array(pos))
	scalar_store.append(Swimmers.Meshed())

#Trying with UpdateSlice to see difference 

Swimmers.SetIC(initial_conditions)

#Fixed the initialization
pos_store2 = list([pos2[:,:]])
pos_store2 = list([np.array(pos2)])
scalar_store2 = list([Swimmers.Meshed()])
for m in range(0,K):
	Swimmers.UpdateSlice(Swimmers.scalar,pos2,vel2)
	pos_store2.append(np.array(pos2))
	scalar_store2.append(Swimmers.Meshed())
relErr = []
for m in range(1,K):
	relErr = np.append(relErr,(sum(scalar_store2[m]) - sum(scalar_store[m]))/(sum(scalar_store[m])))
np.save('ScalarRegSmallPert_CNewI',scalar_store)
np.save('ScalarSliceSmallPert_CNewI',scalar_store2)
print('Relative Error: {0}'.format(relErr))
