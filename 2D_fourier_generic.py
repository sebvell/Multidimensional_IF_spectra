#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 14:07:53 2017
This program calculates the spike-train power spectrum for the generalized two-dimensional integrate and fire neuron from the numerical solution of
the Fokker-Planck equation in Fourier domain. 

@author: Sebastian Vellmer
sebastian.vellmer@bccn-berlin.de
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sci
import scipy as scl
import scipy.optimize as sp
import timeit as time

#options: 
iterative=False 
direct=True
AdEx=False

"""
iterative: for the iterative solution, the power spectrum is only calculated at 0 and the firing rate and the input parameter for the next 
iteration are calcualted
direct: determines the algorithm to solve the sparse linear systems. True=LU decomposition, False=BicGradStab procedure
AdEx: set the neuron model to either AdEx (True) or LIF (False)

"""
#settings for a nice plot
plt.rc('font',**{'family':'serif','serif':['Times']},size=14)
plt.rc('text', usetex=True)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.figsize'] = 5.5,4



#plotting routines: mapP plots the 2D probability density, plot_proj the two projetcions
def mapP(p):
    P_map=np.zeros([N_ele,N_ele])
    pp=p
    pp[pp<0]=0
    for i in range(N_ele):
        P_map[i,:]=np.real(pp[i*N_ele:(i+1)*N_ele])
    plt.figure()
    plt.imshow(P_map[:,:]*1000, interpolation='none', aspect='auto',extent=([v[0],v[-1],a[0],a[-1]]), origin='lower')
    plt.xlabel('v [mV]')
    plt.ylabel('a [mV]')
    cbar=plt.colorbar()
    cbar.set_label(r'$P_0$(v,a) [1/mV$^2$] $\cdot 10^{-3}$')
    plt.plot([20,20],[a[0],a[-1]],color='red',linewidth=4.0)
    plt.plot([0,0],[a[0],a[-1]],':',color='white',linewidth=2.0)
    plt.xlim([v[0],vth])
    plt.tight_layout()
    return

def plot_proj(p):
    plt.figure()
    Pv=np.zeros(N_ele)
    Pa=np.zeros(N_ele)
    for i in range(N_ele):
        Pv[i]=np.sum(P0[i+aloc]*da*dv)
    plt.plot(v,Pv)
    plt.xlabel('voltage [mV]')
    plt.figure()
    for i in range(N_ele):
        Pa[i]=np.sum(P0[aloc[i]+vloc])*dv*da
    plt.plot(a,Pa)
    plt.xlabel('a [mV]')

#Neuron params
tau_m=0.02
vth=20.0
vr=0.0
tref=0.002
beta=4.
mu=15.
vref=50

#auxiliary params
tau_a=0.005
beta1=-2.74
beta2=0
delta_a=0.
A=0.0

if AdEx==True:
    vth=3
    vT=-3
    Delta_T=2
    def f_fun(v):
        return -v+Delta_T*np.exp((v-vT)/Delta_T)
    def F_fun(a):
        return -a+mu
else:
    def f_fun(v):
        return -v+mu
    def F_fun(a):
        return a

def g_fun(a):
    return -a
def G_fun(v):
    return A*v

    



#######################################################################################
#Fokker-Planck Operator
N_ele=1000
aloc=np.array(np.linspace(0,(N_ele-1)*N_ele,N_ele),dtype=int)
vloc=np.array(np.linspace(0,(N_ele-1),N_ele),dtype=int)

#limits have to be set manually
if beta1**2+beta2**2>0:
    a0=-np.abs(beta1)*np.sqrt(10./tau_a)
    aN_1=np.abs(beta1)*np.sqrt(10./tau_a)+10*delta_a
else:
    a0=-2
    aN_1=5*delta_a
v0=-40
if v0>0:
    v0=-1

#a-space
a=np.linspace(a0,aN_1,N_ele)
da=a[1]-a[0]

#v-space
dv=(vth-v0)/(N_ele)
v=np.linspace(v0,vth-dv,N_ele)
nr=int(np.round((vr-v[0])/dv))



########################################################################################################
print('initialize operator')
#Fokker_Planck Operator
start=time.default_timer()
vv=beta**2/tau_m**2/2/dv**2
aa=(beta1**2+beta2**2)/tau_a**2/2/da**2
diag=-2*vv-2*aa
c1=-F_fun(a)/2./tau_m/dv
c2=-f_fun(v)/2./tau_m/dv
c3=beta*beta1/tau_m/tau_a/4./da/dv
c4=-g_fun(a)/tau_a/2./da
c5=-G_fun(v)/2./tau_a/da

L=sci.lil_matrix((N_ele**2,N_ele**2))
R=sci.lil_matrix((N_ele**2,N_ele**2))


L.setdiag(diag)
for j in range(N_ele-1):
    L[vloc[j]+aloc,vloc[j+1]+aloc]=c1+c2[j+1]+vv
    L[vloc+aloc[j],vloc+aloc[j+1]]=aa+c4[j+1]+c5
    L[vloc[1:]+aloc[j],vloc[1:]-1+aloc[j+1]]=-c3
    L[vloc[:-1]+aloc[j],vloc[:-1]+1+aloc[j+1]]=+c3
for j in range(1,N_ele):
    L[vloc[j]+aloc,vloc[j-1]+aloc]=-c1-c2[j-1]+vv
    L[vloc+aloc[j],vloc+aloc[j-1]]=aa-c4[j-1]-c5
    L[vloc[1:]+aloc[j],vloc[1:]-1+aloc[j-1]]=+c3
    L[vloc[:-1]+aloc[j],vloc[:-1]+1+aloc[j-1]]=-c3

#Absorbing boundary condition to reset
#R[nr+aloc,aloc+N_ele-1]=vv

n_delta_a=int(delta_a/da)
part_a=delta_a/da-n_delta_a
nr=int((vr-v[0])/dv)
part_r=(vr-v[0])/dv-nr
if n_delta_a==0:
    R[nr+aloc,aloc+N_ele-1]=vv*(1-part_a)*(1-part_r)
    R[nr+1+aloc,aloc+N_ele-1]=vv*(1-part_a)*part_r
    R[nr+aloc[1:],aloc[:-1]+N_ele-1]=vv*part_a*(1-part_r)
    R[nr+1+aloc[1:],aloc[:-1]+N_ele-1]=vv*part_a*part_r
else:
    R[nr+aloc[n_delta_a:],aloc[:-n_delta_a]+N_ele-1]=vv*(1-part_a)*(1-part_r)
    R[nr+1+aloc[n_delta_a:],aloc[:-n_delta_a]+N_ele-1]=vv*(1-part_a)*part_r
    R[nr+aloc[1+n_delta_a:],aloc[:-n_delta_a-1]+N_ele-1]=vv*part_a*(1-part_r)
    R[nr+1+aloc[1+n_delta_a:],aloc[:-n_delta_a-1]+N_ele-1]=vv*part_a*part_r

#Diffusion of a during refractory period
if tref>0:    
    if aa==0:
        x_ref=np.zeros([N_ele,N_ele])
        fac=np.exp(-tref/tau_a)
        delta_a=A*vref*(1-fac)
        for j in range(N_ele):
            n_a=(a[j]*fac+delta_a-a[0])/da
            part_a=n_a-int(n_a)
            x_ref[int(n_a),j]=1-part_a
            x_ref[int(n_a)+1,j]=part_a        
    else:
        N_ts=int(4*aa*tref)
        c5=-G_fun(vref)/2./tau_a/da
        x_ref=np.matrix(np.zeros([N_ele,N_ele]))
        x_ref[vloc[:-1],vloc[1:]]=(aa+c4[1:]+c5)*tref/N_ts
        x_ref[vloc[1:],vloc[:-1]]=(aa-c4[:-1]-c5)*tref/N_ts
        x_ref[vloc[:],vloc[:]]=(-2*aa*tref/N_ts+1)*np.ones(N_ele)
        x_ref=x_ref**N_ts
    x_reff=sci.lil_matrix((N_ele**2,N_ele**2))
    for j in range(N_ele):
        x_reff[nr+aloc[j],nr+aloc]=x_ref[vloc[j],vloc]
        x_reff[nr+1+aloc[j],nr+1+aloc]=x_ref[vloc[j],vloc]
    R=x_reff.dot(R)
    del x_reff
    del x_ref
    
#stationary solution
one=sci.lil_matrix((N_ele**2,N_ele**2))
one[:,int(N_ele**2/2+nr)]=one[:,int(N_ele**2/2+nr)]+(np.zeros([N_ele**2,1])+1)
one.tocsc()
L = L.tocsc()
R=R.tocsc()

end=time.default_timer()
print(end-start)

print('calculate stationary solution')
start=time.default_timer()
if direct==True:
    P=sci.linalg.spsolve(L+R+one,np.zeros(N_ele**2)+1,use_umfpack=False)
    end=time.default_timer()
else:
    P=sci.linalg.spilu(L+R+one,fill_factor=30,drop_tol=1e-4)
    end=time.default_timer()
    print(end-start)
    P=sci.linalg.LinearOperator((N_ele**2,N_ele**2),P.solve)
    end=time.default_timer()
    print(end-start)
    P=sci.linalg.bicgstab((L+R+one),np.zeros(N_ele**2)+1,M=P)[0]
end=time.default_timer()
print('time: '+str(end-start))
del one
P=P/np.sum(P)

print('error '+str(np.sum(P[P<0])))
p_v0=0
p_a0=0
p_aN=0
for i in range(int(N_ele/10)):
    p_v0+=np.sum(P[aloc+vloc[i]])
    p_a0+=np.sum(P[aloc[i]+vloc])
    p_aN+=np.sum(P[aloc[-1-i]+vloc])
print(p_v0)
print(p_a0)
print(p_aN)
P=P/dv/da
Ps=R.dot(P)
Ps[Ps<0]=0
rate=np.sum(Ps)*dv*da
Ps=Ps/np.sum(Ps)/dv/da
rate=1/(tref+1/rate)
P0=P*(1-tref*rate)
print('done')
print('rate: '+str(rate))

if iterative==True:
    n_omega=2
    omega=np.linspace(0,2*np.pi*rate,n_omega)
else:
    n_omega=21
    omega_max=1000*2*np.pi
    omega=np.append(0,np.exp(np.linspace(np.log(np.pi),np.log(omega_max),n_omega-1)))
m=np.zeros(n_omega)
for i in range(n_omega):
    right=np.zeros(N_ele,dtype=complex)
    if omega[i]==0:
        right=(1-rate*tref)*Ps-P0
        if direct==True:
            P=sci.linalg.spsolve(-L-R,right,use_umfpack=False)
        else:
            P=sci.linalg.spilu((-L-R).tocsc())
            P=sci.linalg.LinearOperator((N_ele**2,N_ele**2),P.solve)
            P=sci.linalg.bicgstab((-L-R),right,M=P)[0]
        P=P+(-np.sum(P)*dv*da+tref*(tref*rate*0.5-1-dv*da*np.sum(R.dot(P))))*P0     
    else:
        eref=np.exp(-1j*omega[i]*tref)
        Op=sci.lil_matrix((N_ele**2,N_ele**2),dtype=complex)
        Op.setdiag(1j*omega[i])
        Op.tocsr()
        Op=Op-L-eref*R
        right=np.zeros(N_ele,dtype=complex)
        right=(eref+rate*1j/omega[i]*(1-eref))*Ps-P0
        if direct==True:
            P=sci.linalg.spsolve(Op,right,use_umfpack=False)
        else:
            P=sci.linalg.spilu((Op).tocsc())
            P=sci.linalg.LinearOperator((N_ele**2,N_ele**2),P.solve)
            P=sci.linalg.bicgstab(Op,right,M=P)[0]
        if omega[i]<=2*np.pi:
            P=P+(-np.sum(P)*dv*da+tref*(tref*rate*0.5-1-dv*da*np.sum(R.dot(P))))*P0

    m[i]=2*np.real(np.sum(R.dot(P))*dv*da)
    np.savez(str(i)+'.npz',m=rate*(1+m[i]),f=omega[i]/2/np.pi)
    print(i)
s=rate*(1+m)
f=omega/2/np.pi
