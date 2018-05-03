# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 21:09:54 2018

@author: rossi
"""

import matplotlib.pyplot as plt
import matplotlib.animation

def plotScene(Swimmers,pos):
    plt.pcolormesh(Swimmers.xm,Swimmers.ym,Swimmers.Meshed())
    plt.plot(pos[:,0],pos[:,1],'ro')
    plt.colorbar()
    return None
    
def animate(k):
    arr = scalar_store[k]
    arr = arr[:-1, :-1]
    field.set_array(arr.ravel())
    plt.title('Frame {0:d}'.format(k))
    dots.set_data(pos_store[k][:,0],pos_store[k][:,1])

    return field,dots,

def buildMovie(Swimmers,scalar_store):
    fig   = plt.figure()
    ax    = plt.subplot(1,1,1)
    field = ax.pcolormesh(Swimmers.xm,Swimmers.ym,scalar_store[1])
    field.set_clim(0,0.08)
    dots, = ax.plot([], [], 'ro')
    fig.colorbar(field)
    
    anim = matplotlib.animation.FuncAnimation(fig,animate,frames=range(0,len(scalar_store),2),
                                              interval=50,blit=False,repeat=True)
    return anim
