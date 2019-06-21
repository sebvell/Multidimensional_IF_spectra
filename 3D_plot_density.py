#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:09:47 2017

This program plots the stationary density of a leaky-integrate-and-fire neuron driven by a two-dimensional Ornstein-Uhlenbeck process and its projections.
As input, the stationary density P0.txt and a parameter file params.txt is required. 

@author: Sebastian Vellmer
"""

import numpy as np
import matplotlib.pyplot as plt
import timeit as time
plt.rc('font',**{'family':'serif','serif':['Times']},size=14)
plt.rc('text', usetex=True)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['figure.figsize'] = 7.5,5


def mapP(p):
    pp=p
    for j in range(10):
        P_map=np.zeros([N_a1,N_a2])
        for i in range(N_a1):
            P_map[i,:]=np.real(pp[a2loc+a1loc[i]+vloc[int((j+1)*N_v/10-1)]])
        plt.figure()
        plt.imshow(P_map[:,:], interpolation='none',cmap='gist_ncar', aspect='auto',extent=([a1[0],a1[-1],a2[0],a2[-1]]), origin='lower')
        plt.xlabel('a1 [mV]')
        plt.ylabel('a2 [mV]')
        cbar=plt.colorbar()
        cbar.set_label(r'$P_0$(v,a1,a2) [1/mV$^2$]')
        plt.tight_layout()
    return

def plot_proj(p):
    plt.figure()
    Pv=np.zeros(N_v)
    Pa2=np.zeros(N_a2)
    Pa1=np.zeros(N_a1)
    for i in range(N_v):
        Pv[i]=np.sum(P0[i+a2loc+a1loc]*da2*da1)
    plt.plot(v,Pv)
    plt.xlabel('voltage [mV]')
    plt.figure()
    for i in range(N_a1):
        for j in range(N_v):
            Pa1[i]+=np.sum(P0[a1loc[i]+a2loc+vloc[j]])*dv*da2
    plt.plot(a1,Pa1)
    plt.ylabel('a1')
    plt.figure()
    for i in range(N_a2):
        for j in range(N_v):
            Pa2[i]+=np.sum(P0[vloc[j]+a2loc[i]+a1loc])*dv*da1
    plt.plot(a2,Pa2)
    plt.ylabel('a2')
    
    
P0=np.genfromtxt('P0.txt', unpack=True)
par=np.genfromtxt('params.txt', unpack=True)
N_v=int(par[0])
N_a1=int(par[1])
N_a2=int(par[2])
v0=par[3]
vth=par[4]
a1_0=par[5]
a1_N_1=par[6]
a2_0=par[7]
a2_N_1=par[8]

a2loc=np.array(np.linspace(0,(N_a2-1)*N_a1*N_v,N_a2),dtype=int)
a1loc=np.array(np.linspace(0,(N_a1-1)*N_v,N_a1),dtype=int)
vloc=np.array(np.linspace(0,(N_v-1),N_v),dtype=int)


#y-space
a1=np.linspace(a1_0,a1_N_1,N_a1)
da1=a1[1]-a1[0]

#s-space
a2=np.linspace(a2_0,a2_N_1,N_a2)
da2=a2[1]-a2[0]

#v-spaces
dv=(vth-v0)/(N_v+1)
v=np.linspace(v0,vth-dv,N_v)
dv=v[1]-v[0]

mapP(P0)
plot_proj(P0)



