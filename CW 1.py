# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:09:13 2023

@author: yashi
"""
#%%

import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(__file__)

#%%

# define formula to solve a NxN matrix of coefficients, A, and property, T, and source terms, d, of length N

def TDMA (A, T, d):
    F = np.zeros(len(T))
    
    P = np.zeros(len(T))
    
    Q = np.zeros(len(T))
    
    #check all the lengths are the same.
    if len(A) == len(T) and len(A) == len (d):
        
        # assume dirichlet boundary conditions
        
        # determine coefficients of P and Q
        
        for i in range (0, len(A)):
            
            # at i = 1
            if i == 0:
                      
                P[i] = 0
                Q[i] = T[i]
                F[i] = T[i]
                
            # at i = N
            elif i == (len(A)-1):
                
                P[i] = 0
                Q[i] = T[i]
                F[i] = T[i]
            
            # for middle values of i
            else:
                
                P[i] = (-A[i][i+1])/(A[i][i]+A[i][i-1]*P[i-1])
                Q[i] = (d[i]-A[i][i-1]*Q[i-1])/(A[i][i]+A[i][i-1]*P[i-1])
            
        # determine values of field T not including boundary conditions
        for i in range(len(T)-2,0,-1):
            
            F[i] = P[i]*F[i+1] + Q[i]
        
        
        return F
    
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
    
    F = np.zeros(len(P))
    F[0] = P[0]
    F[len(P)-1] = P[len(P)-1]
    #determine global peclet number
    Pe = (R*u*x[N-1])/G
    
    for i in range(1,N-1):
        F[i] = P[0] + ((np.exp(x[i]*Pe/x[len(x)-1])-1)/(np.exp(Pe)-1))*(P[len(P)-1]-P[0])
        
    return F
    
#%%

# determine accuracy of a model

def NumAcc (P, A, N):
    
    #check lengths
    E = 0
    if len(P) == len(A):
        
        
        for i in range(0,len(P)):
            
            E += 100*(P[i]-A[i])/(A[i]*N)
            
        #E = 100*E/N
        return E
        
    else:
        return (print('ERROR: Dimensions not consistent.'))
        
#%%
#fixed parameters

u = np.linspace(1, 50, 11, endpoint = 'True') # fluid velocity range

Gamma_phi = 0.5 #diffusion coefficent

rho = 0.5 # fluid density

L=1 #maximum length of 1-D domain

N = [11, 51, 101, 501, 1001, 5001, 10001, 50001] # number of nodes along 1-D length range


results_dir = os.path.join(script_dir, 'rho=%.2f_'%rho+'gamma=%.2f/'%Gamma_phi)
os.makedirs(results_dir, exist_ok=True)# compute solution

##CDS

### change value of u

for v in u:
    
#### change value of N
    delx = []
    Acc = []

    for n in N:
        
        #create grid
        x = np.linspace(0, L, n)
        
        phi = np.zeros(len(x))

        phi[0] = 100

        phi[len(x)-1] = 20

        S = np.zeros(len(phi))
        
        #determine peclet numbers
        
        #global peclet
        Pe_G = (rho*u*L)/Gamma_phi
        
        #local peclet
        Pe_x = (rho*u*(L/n))/Gamma_phi
        
        # find the coefficients
        
        A = CDS(phi, n, x, Gamma_phi, rho, v)
        
        # solve for phi
        phi_num = TDMA(A, phi, S)
        
        # plot values of phi as contour/gradient
        fig, ax = plt.subplots()
        ax.plot(x, phi_num)
        
        extent = min(x), max(x), min(phi_num), max(phi_num)
        pcm = ax.imshow(np.expand_dims(phi_num, axis = 0), interpolation=None, aspect='auto', cmap = 'viridis', extent = extent)
        fig.colorbar(pcm)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$\\phi$')
        #ax.set_title('CDS: 1-D Heat Diffusion \n$u = %d$, '%v+'$N = %d$, '%n+'$\\rho = %.2f$, '%rho+'$\\Gamma_\\phi = %.2f$' %Gamma_phi)
        plt.show()
        fig.savefig(os.path.join(results_dir,'CDS_1-D_Heat_Diffusion_u=%d_'%v+'N=%d.png'%n))
        
        # determine analytical solution
        phi_ana = Analytical(phi, n, x, Gamma_phi, rho, v)
        
        # plot numerical and analytical solution with global and local peclet number
        fig, ax = plt.subplots()
        ax.scatter(x, phi_num, color = 'b')
        ax.plot(x,phi_num, color = 'b', linestyle = 'dashed')
       
        ax.plot(x, phi_ana, color = 'r')
        
        ax.set_xlabel('$x$')
        ax.set_ylabel('$\\phi$')
        #ax.set_title('CDS: Numerical vs Analytical \n$u = %d$, '%v+'$N = %d$, '%n+'$\\rho = %.2f$, '%rho+'$\\Gamma_\\phi = %.2f$' %Gamma_phi+'\nGlobal $Pe = %.3f$, '%Pe_G+ 'Local $Pe = %.3f$' %Pe_x)
        plt.show()
        fig.savefig(os.path.join(results_dir,'CDS_Numerical_vs_Analytical_u=%d_'%v+'N=%d.png'%n))
        # determine acuracy for grid spacing
        
        delx = delx + [(L/n)]
        Acc = Acc + [NumAcc(phi_num, phi_ana, n)]
        
        
    # plot error against grid spacing delx with convective flux value
    fig, ax = plt.subplots()
    ax.plot(delx, Acc, color = 'g')
    ax.set_xlabel('$\\delta x$')
    ax.set_ylabel('Error (%)')
    #ax.set_title('CDS: Gridspace Error \n$u= %d$, ' %v+'$\\rho = %.2f$, '%rho+'$\\Gamma_\\phi = %.2f$' %Gamma_phi)
    fig.savefig(os.path.join(results_dir,'CDS_Gridspace_Error_u=%d.png'%v))
    plt.show()

##UDS

### change value of u

for v in u:
    
#### change value of N
    delx = []
    Acc = []

    for n in N:
        
        #create grid
        x = np.linspace(0, L, n)
        
        phi = np.zeros(len(x))

        phi[0] = 100

        phi[len(x)-1] = 20

        S = np.zeros(len(phi))
        
        #determine peclet numbers
        
        #global peclet
        Pe_G = (rho*u*L)/Gamma_phi
        
        #local peclet
        Pe_x = (rho*u*(L/n))/Gamma_phi
        
        # find the coefficients
        
        A = UDS(phi, n, x, Gamma_phi, rho, v)
        
        # solve for phi
        phi_num = TDMA(A, phi, S)
        
        # plot values of phi as contour/gradient
        fig, ax = plt.subplots()
        ax.plot(x, phi_num)
        
        extent = min(x), max(x), min(phi_num), max(phi_num)
        pcm = ax.imshow(np.expand_dims(phi_num, axis = 0), interpolation=None, aspect='auto', cmap = 'viridis', extent = extent)
        fig.colorbar(pcm)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$\\phi$')
        #ax.set_title('UDS: 1-D Heat Diffusion \n$u = %d$, '%v+'$N = %d$, '%n+'$\\rho = %.2f$, '%rho+'$\\Gamma_\\phi = %.2f$' %Gamma_phi)
        plt.show()
        fig.savefig(os.path.join(results_dir,'UDS_1-D_Heat_Diffusion_u=%d_'%v+'N=%d.png'%n))
        
        # determine analytical solution
        phi_ana = Analytical(phi, n, x, Gamma_phi, rho, v)
        
        # plot numerical and analytical solution with global and local peclet number
        fig, ax = plt.subplots()
        ax.scatter(x, phi_num, color = 'b')
        ax.plot(x,phi_num, color = 'b', linestyle = 'dashed')
       
        ax.plot(x, phi_ana, color = 'r')
        
        ax.set_xlabel('$x$')
        ax.set_ylabel('$\\phi$')
        #ax.set_title('UDS: Numerical vs. Analytical \n$u = %d$, '%v+'$N = %d$, '%n+'$\\rho = %.2f$, '%rho+'$\\Gamma_\\phi = %.2f$' %Gamma_phi+'\nGlobal $Pe = %.3f$, '%Pe_G+ 'Local $Pe = %.3f$' %Pe_x)
        plt.show()
        fig.savefig(os.path.join(results_dir,'UDS_Numerical_vs_Analytical_u=%d_'%v+'N=%d.png'%n))
        # determine acuracy for grid spacing
        
        delx = delx + [(L/n)]
        Acc = Acc + [NumAcc(phi_num, phi_ana, n)]
        
        
    # plot error against grid spacing delx with convective flux value
    fig, ax = plt.subplots()
    ax.plot(delx, Acc, color = 'g')
    ax.set_xlabel('$\\delta x$')
    ax.set_ylabel('Error (%)')
    #ax.set_title('UDS: Gridspace Error \n$u= %d$, ' %v+'$\\rho = %.2f$, '%rho+'$\\Gamma_\\phi = %.2f$' %Gamma_phi)
    fig.savefig(os.path.join(results_dir,'UDS_Gridspace_Error_u=%d.png'%v))
    plt.show()

##PLDS

### change value of u

for v in u:
    
#### change value of N
    delx = []
    Acc = []

    for n in N:
        
        #create grid
        x = np.linspace(0, L, n)
        
        phi = np.zeros(len(x))

        phi[0] = 100

        phi[len(x)-1] = 20

        S = np.zeros(len(phi))
        
        #determine peclet numbers
        
        #global peclet
        Pe_G = (rho*u*L)/Gamma_phi
        
        #local peclet
        Pe_x = (rho*u*(L/n))/Gamma_phi
        
        # find the coefficients
        
        A = PLDS(phi, n, x, Gamma_phi, rho, v)
        
        # solve for phi
        phi_num = TDMA(A, phi, S)
        
        # plot values of phi as contour/gradient
        fig, ax = plt.subplots()
        ax.plot(x, phi_num)
        
        extent = min(x), max(x), min(phi_num), max(phi_num)
        pcm = ax.imshow(np.expand_dims(phi_num, axis = 0), interpolation=None, aspect='auto', cmap = 'viridis', extent = extent)
        fig.colorbar(pcm)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$\\phi$')
        #ax.set_title('PLDS: 1-D Heat Diffusion \n$u = %d$, '%v+'$N = %d$, '%n+'$\\rho = %.2f$, '%rho+'$\\Gamma_\\phi = %.2f$' %Gamma_phi)
        plt.show()
        fig.savefig(os.path.join(results_dir,'PLDS_1-D_Heat_Diffusion_u=%d_'%v+'N=%d.png'%n))
        
        # determine analytical solution
        phi_ana = Analytical(phi, n, x, Gamma_phi, rho, v)
        
        # plot numerical and analytical solution with global and local peclet number
        fig, ax = plt.subplots()
        ax.scatter(x, phi_num, color = 'b')
        ax.plot(x,phi_num, color = 'b', linestyle = 'dashed')
       
        ax.plot(x, phi_ana, color = 'r')
        
        ax.set_xlabel('$x$')
        ax.set_ylabel('$\\phi$')
        #ax.set_title('PLDS: Numerical vs. Analytical \n$u = %d$, '%v+'$N = %d$, '%n+'$\\rho = %.2f$, '%rho+'$\\Gamma_\\phi = %.2f$' %Gamma_phi+'\nGlobal $Pe = %.3f$, '%Pe_G+ 'Local $Pe = %.3f$' %Pe_x)
        plt.show()
        fig.savefig(os.path.join(results_dir,'PLDS_Numerical_vs_Analytical_u=%d_'%v+'N=%d.png'%n))
        # determine acuracy for grid spacing
        
        delx = delx + [(L/n)]
        Acc = Acc + [NumAcc(phi_num, phi_ana, n)]
        
        
    # plot error against grid spacing delx with convective flux value
    fig, ax = plt.subplots()
    ax.plot(delx, Acc, color = 'g')
    ax.set_xlabel('$\\delta x$')
    ax.set_ylabel('Error (%)')
    #ax.set_title('PLDS: Gridspace Error \n$u= %d$, ' %v+'$\\rho = %.2f$, '%rho+'$\\Gamma_\\phi = %.2f$' %Gamma_phi)
    fig.savefig(os.path.join(results_dir,'PLDS_Gridspace_Error_u=%d.png'%v))
    plt.show()