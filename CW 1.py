# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:09:13 2023

@author: yashi
"""
#%%

import numpy as np

#%%

# define formula to solve a NxN matrix of coefficients, A, and field, T, and source terms, d, of length N

def TDMA (A, T, d):
    
    P = np.zeros(len(T))
    
    Q = np.zeros(len(T))
    
    #check all the lengths are the same.
    if len(A) == len(T) and len(A) == len (d):
        
        # assume dirihlet boundary conditions
        
        # determine coefficients of P and Q
        
        for i in range (0, len(A)):
            
            # at i = 1
            if i == 0:
                
                
                P[i] = 0
                Q[i] = T[i]
                
            # at i = N
            elif i == (len(A)-1):
                
                P[i] = 0
                Q[i] = T[i]
            
            # for middle values of i
            else:
                
                P[i] = (-A[i][i+1])/(A[i][i]+A[i][i-1]*P[i-1])
                Q[i] = (d[i]-A[i][i-1]*Q[i-1])/(A[i][i]+A[i][i-1]*P[i-1])
            
        # determine values of field T not including boundary conditions
        for i in range(len(T)-2,0,-1):
            
            T[i] = P[i]*T[i+1] + Q[i]
        
        
        return T
    
    else:
        return print('ERROR: Dimensions not consistent.')

#%%

# define central difference scheme to return a matrix of coefficients from parameters

def CDS (phi, N, x, Gamma_phi, rho, u):

    A = np.zeros((len(phi),len(phi)))
    
    
    for p in range(0,N): #determine phi between boundaries
            
        # boundary conditions
        
        if p == 0:
            
            #determine delx_E
            delx_E = x[p+1]-x[p]
            
            #determine coeffiients
            a_E = (Gamma_phi/delx_E) - (rho*u)
            
            a_p = a_E
            
            #add coefficients to matrix A
            A[p][p] = a_p
            A[p][p+1] = a_E*(-1)
            
        elif p == (N-1):
            
            #determine delx_W
            delx_W = x[p] - x[p-1]
            
            #determine coeffiients
            a_W = (Gamma_phi/delx_W) + (rho*u)
            
            a_p = a_W
            
            #add coefficients to matrix A
            A[p][p] = a_p
            A[p][p-1] = a_W*(-1)
            
        else:
            
            #determine delx_W, delx_E
            
            delx_W = x[p] - x[p-1]
            
            delx_E = x[p+1]-x[p]
            
            
            #determine coefficients
            
            a_E = (Gamma_phi/delx_E) - (rho*u)
            
            a_W = (Gamma_phi/delx_W) + (rho*u)
            
            a_p = a_E + a_W
        
            #add coefficients to matrix A
            A[p][p] = a_p
            A[p][p+1] = a_E*(-1)
            A[p][p-1] = a_W*(-1)
    
    return A
    
#%%

# define upward differencing model

def UDS (phi, N, x, Gamma_phi, rho, u):
    
    A = np.zeros((len(phi),len(phi)))
    
    for p in range(0,N): #determine phi between boundaries
            
        # boundary conditions
        
        if p == 0:
            
            #determine delx_E
            delx_E = x[p+1]-x[p]
            
            #determine coeffiients
            a_E = (Gamma_phi/delx_E) + max((rho*u*(-1)), 0) 
            
            a_p = a_E
            
            #add coefficients to matrix A
            A[p][p] = a_p
            A[p][p+1] = a_E*(-1)
            
        elif p == (N-1):
            
            #determine delx_W
            delx_W = x[p] - x[p-1]
            
            #determine coeffiients
            a_W = (Gamma_phi/delx_W) + max((rho*u), 0)
            
            a_p = a_W
            
            #add coefficients to matrix A
            A[p][p] = a_p
            A[p][p-1] = a_W*(-1)
            
        else:
            
            #determine delx_W, delx_E
            
            delx_W = x[p] - x[p-1]
            
            delx_E = x[p+1]-x[p]
            
            
            #determine coefficients
            
            a_W = (Gamma_phi/delx_W) + max((rho*u), 0)
            
            a_E = (Gamma_phi/delx_E) + max((rho*u*(-1)), 0)
            
            a_p = a_E + a_W
        
            #add coefficients to matrix A
            A[p][p] = a_p
            A[p][p+1] = a_E*(-1)
            A[p][p-1] = a_W*(-1)
    
    return A

#%%

# define power-law differencing model

def PLDS (phi, N, x, Gamma_phi, rho, u):
    
    A = np.zeros((len(phi),len(phi)))
    
    for p in range(0,N): #determine phi between boundaries
            
        # boundary conditions
        
        if p == 0:
            
            #determine delx_E
            delx_E = x[p+1]-x[p]
            
            #determine locl peclet number
            
            Pe_e = (rho*u*delx_E/Gamma_phi)
            
            #determine coeffiients
            a_E = (Gamma_phi/delx_E)*max((1-0.1*abs(Pe_e))**5,0) + max((rho*u*(-1)), 0) 
            
            a_p = a_E
            
            #add coefficients to matrix A
            A[p][p] = a_p
            A[p][p+1] = a_E*(-1)
            
        elif p == (N-1):
            
            #determine delx_W
            delx_W = x[p] - x[p-1]
            
            #determine locl peclet number
            
            Pe_w = (rho*u*delx_W/Gamma_phi)
            
            #determine coeffiients
            a_W = (Gamma_phi/delx_W)*max((1-0.1*abs(Pe_w))**5,0) + max((rho*u), 0)
            
            a_p = a_W
            
            #add coefficients to matrix A
            A[p][p] = a_p
            A[p][p-1] = a_W*(-1)
            
        else:
            
            #determine delx_W, delx_E
            
            delx_W = x[p] - x[p-1]
            
            delx_E = x[p+1]-x[p]
            
            #determine locl peclet number
            
            Pe_e = (rho*u*delx_E/Gamma_phi)
            
            Pe_w = (rho*u*delx_W/Gamma_phi)
            
            #determine coefficients
            
            a_E = (Gamma_phi/delx_E)*max((1-0.1*abs(Pe_e))**5,0) + max((rho*u*(-1)), 0)
            
            a_W = (Gamma_phi/delx_W)*max((1-0.1*abs(Pe_w))**5,0) + max((rho*u), 0)
            
            a_p = a_E + a_W
        
            #add coefficients to matrix A
            A[p][p] = a_p
            A[p][p+1] = a_E*(-1)
            A[p][p-1] = a_W*(-1)
    
    return A

#%%
#fixed parameters

u = 10 # fluid velocity
Gamma_phi = 0.9 #diffusion coefficent
rho = 0.2 # 

#create grid
L=1 #maximum length of 1-D domain

N = 1000 # number of points along 1-D length

delx = L/(N-1)

x = np.linspace(0, L, N) #creates grid of even spacing

phi = np.zeros(len(x))

phi[0] = 100

phi[len(x)-1] = 20

S = np.zeros(len(phi))

# compute solution

A = CDS(phi, N, x, Gamma_phi, rho, u)      
print (A)
    
phi = TDMA(A, phi, S)

print(phi)
