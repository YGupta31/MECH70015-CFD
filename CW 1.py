# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:09:13 2023

@author: yashi
"""
#%%

import numpy as np
import matplotlib.pyplot as plt


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

def CDS (P, N, x, G, R, u): # P = property , N = nodes, x = Length domain, G = diffusion coefficient, R = density, u = flow velocity 

    A = np.zeros((len(P),len(P)))
    
    
    for p in range(0,N): #determine phi between boundaries
            
        # boundary conditions
        
        if p == 0:
            
            #determine delx_E
            delx_E = x[p+1] - x[p]
            
            #determine coeffiients
            a_E = (G/delx_E) - (R*u)
            
            a_p = a_E
            
            #add coefficients to matrix A
            A[p][p] = a_p
            A[p][p+1] = a_E*(-1)
            
        elif p == (N-1):
            
            #determine delx_W
            delx_W = x[p] - x[p-1]
            
            #determine coeffiients
            a_W = (G/delx_W) + (R*u)
            
            a_p = a_W
            
            #add coefficients to matrix A
            A[p][p] = a_p
            A[p][p-1] = a_W*(-1)
            
        else:
            
            #determine delx_W, delx_E
            
            delx_W = x[p] - x[p-1]
            
            delx_E = x[p+1] - x[p]
            
            
            #determine coefficients
            
            a_E = (G/delx_E) - (R*u)
            
            a_W = (G/delx_W) + (R*u)
            
            a_p = a_E + a_W
        
            #add coefficients to matrix A
            A[p][p] = a_p
            A[p][p+1] = a_E*(-1)
            A[p][p-1] = a_W*(-1)
    
    return A
    
#%%

# define upward differencing model

def UDS (P, N, x, G, R, u):
    
    A = np.zeros((len(P),len(P)))
    
    for p in range(0,N): #determine phi between boundaries
            
        # boundary conditions
        
        if p == 0:
            
            #determine delx_E
            delx_E = x[p+1] - x[p]
            
            #determine coeffiients
            a_E = (G/delx_E) + max((R*u*(-1)), 0) 
            
            a_p = a_E
            
            #add coefficients to matrix A
            A[p][p] = a_p
            A[p][p+1] = a_E*(-1)
            
        elif p == (N-1):
            
            #determine delx_W
            delx_W = x[p] - x[p-1]
            
            #determine coeffiients
            a_W = (G/delx_W) + max((R*u), 0)
            
            a_p = a_W
            
            #add coefficients to matrix A
            A[p][p] = a_p
            A[p][p-1] = a_W*(-1)
            
        else:
            
            #determine delx_W, delx_E
            
            delx_W = x[p] - x[p-1]
            
            delx_E = x[p+1] - x[p]
            
            
            #determine coefficients
            
            a_W = (G/delx_W) + max((R*u), 0)
            
            a_E = (G/delx_E) + max((R*u*(-1)), 0)
            
            a_p = a_E + a_W
        
            #add coefficients to matrix A
            A[p][p] = a_p
            A[p][p+1] = a_E*(-1)
            A[p][p-1] = a_W*(-1)
    
    return A

#%%

# define power-law differencing model

def PLDS (P, N, x, G, R, u):
    
    A = np.zeros((len(P),len(P)))
    
    for p in range(0,N): #determine phi between boundaries
            
        # boundary conditions
        
        if p == 0:
            
            #determine delx_E
            delx_E = x[p+1] - x[p]
            
            #determine locl peclet number
            
            Pe_e = (R*u*delx_E/G)
            
            #determine coeffiients
            a_E = (G/delx_E)*max((1-0.1*abs(Pe_e))**5,0) + max((R*u*(-1)), 0) 
            
            a_p = a_E
            
            #add coefficients to matrix A
            A[p][p] = a_p
            A[p][p+1] = a_E*(-1)
            
        elif p == (N-1):
            
            #determine delx_W
            delx_W = x[p] - x[p-1]
            
            #determine locl peclet number
            
            Pe_w = (R*u*delx_W/G)
            
            #determine coeffiients
            a_W = (G/delx_W)*max((1-0.1*abs(Pe_w))**5,0) + max((R*u), 0)
            
            a_p = a_W
            
            #add coefficients to matrix A
            A[p][p] = a_p
            A[p][p-1] = a_W*(-1)
            
        else:
            
            #determine delx_W, delx_E
            
            delx_W = x[p] - x[p-1]
            
            delx_E = x[p+1] - x[p]
            
            #determine locl peclet number
            
            Pe_e = (R*u*delx_E/G)
            
            Pe_w = (R*u*delx_W/G)
            
            #determine coefficients
            
            a_E = (G/delx_E)*max((1-0.1*abs(Pe_e))**5,0) + max((R*u*(-1)), 0)
            
            a_W = (G/delx_W)*max((1-0.1*abs(Pe_w))**5,0) + max((R*u), 0)
            
            a_p = a_E + a_W
        
            #add coefficients to matrix A
            A[p][p] = a_p
            A[p][p+1] = a_E*(-1)
            A[p][p-1] = a_W*(-1)
    
    return A

#%%

# compare to analytical solution

def Analytical (P, N, x, G, R, u):
    
    #determine global peclet number
    
    Pe = (R*u*x[N-1])/G
    
    for i in range(1,N-1):
        P[i] = P[0] + ((np.exp(x[i]*Pe/x[len(x)-1])-1)/(np.exp(Pe)-1))*(P[len(P)-1]-P[0])
        
        return P
    
#%%

# determine accuracy of a model

def NumAcc (P, A, N):
    
    #check lengths
    
    if len(P) == len(A):
        
        E = 0
        
        for i in range(0,len(P)):
            
            E = E + abs((P[i]-A[i])/A[i])
            
        E = 100*E/N
        
    else:
        return (print('ERROR: Dimensions not consistent.'))
        
#%%
#fixed parameters

u = np.linspace(1, 10, 2, endpoint = 'True') # fluid velocity range
Gamma_phi = 0.6 #diffusion coefficent
rho = 0.2 # 
L=1 #maximum length of 1-D domain
#create grid


N = [11, 51, 101]#, 501, 1001, 5001, 10001, 50001] # number of nodes along 1-D length range

#delx = L/(N-1)

#x = np.linspace(0, L, N) #creates grid of even spacing

#phi = np.zeros(len(x))

#phi[0] = 100

#phi[len(x)-1] = 20

#S = np.zeros(len(phi))

# compute solution

##CDS

### change value of u

for v in u:
    
#### change value of N
    delx = []
    Acc = []

    for n in N:
        
        x = np.linspace(0, L, n)
        
        phi = np.zeros(len(x))

        phi[0] = 100

        phi[len(x)-1] = 20

        S = np.zeros(len(phi))
        
        # find the coefficients
        
        A = CDS(phi, n, x, Gamma_phi, rho, v)
        
        # solve for phi
        
        phi_num = TDMA(A, phi, S)
        
        # plot values of phi as contour/gradient
        plt.plot(x, phi_num)
        
        extent = min(x), max(x), min(phi), max(phi)
        plt.imshow(np.expand_dims(phi, axis = 0), interpolation=None, aspect='auto', cmap = 'viridis', extent = extent)
        plt.colorbar()
        plt.show()
        # determine analytical solution
        
        phi_ana = Analytical(phi, n, x, Gamma_phi, rho, v)
        
        # plot numerical and analytical solution with global and local peclet number
        
        plt.plot(x, phi_ana, color = 'r')
        plt.plot(x, phi_num, color = 'b')
        plt.show()
        
        # determine acuracy for grid spacing
        
        delx = delx + (L/n)
        Acc = Acc + [NumAcc(phi_num, phi_ana, n)]
        
        
    # plot error against grid spacing delx with convective flux value
    
    plt.plot(delx, Acc)
    plt.show()

##UDS

##PLDS
