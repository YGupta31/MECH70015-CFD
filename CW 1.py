# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:09:13 2023

@author: yashi
"""
#%%

import numpy as np


#%%
#fixed parameters

u = 10 # fluid velocity
Gamma_phi = 1 #diffusion coefficent
rho = 1 # 

#create grid
L=1 #maximum length of 1-D domain

N = 11 # number of points along 1-D length

delx = L/(N-1)

x = np.linspace(0, L, N) #creates grid of even spacing

phi = np.zeros(len(x))

phi[0] = 100

phi[len(x)-1] = 20


#%%

# Apply central difference scheme

for p in range(1,N-1): #determine phi between boundaries

    #determine delx_W, delx_E
    
    delx_W = delx*p
    
    delx_E = L-delx_W
    
    delx_en = delx/2
    
    delx_ep = delx_E - delx_en
    
    delx_wn = delx/2
    
    delx_wp = delx_W - delx_wn
    
    f_e = delx_en/delx_ep
    
    f_w = delx_wn/delx_wp
    
    #determine coeffiient
    
    a_E = (Gamma_phi/delx_E) - (rho*u*f_e)
    
    a_W = (Gamma_phi/delx_W) + (rho*u*(1-f_w))
    
    a_p = a_E + a_W + (rho*u-rho*u)
    
    phi[p] = ((phi[len(x)-1]*a_E) + (phi[0]*a_E))
    
    print(phi)
        
    