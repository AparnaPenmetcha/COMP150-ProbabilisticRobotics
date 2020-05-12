# Kalman Filters Homework 1 
# Aparna Penmetcha 
# Problem 2.2

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

A_t = [[1, 1],
      [0, 1]]

A_t = np.array(A_t)

B_t = 0
u_t = 0

sigma_w = 1 

C_t = [[1, 0]] 
C_t = np.array(C_t)

Q_t = 8.0 #sigma^2 (covariance of noise)

R_t = sigma_w**2*(np.matmul(np.array([[0.5],[1]]),np.array([[0.5,1]])))

num_timesteps = 2
mu_bar = np.zeros((2,num_timesteps))
cov_bar = np.zeros((num_timesteps,2,2))
mu_t = np.zeros((2,num_timesteps)) 
mu_t[:,0] = np.array([0,0])
cov_t = np.zeros((num_timesteps,2,2))
cov_t[0,:,:] = np.array([[21, 8], #covariance at four 
                        [8, 4]]) 

z_t = np.zeros((num_timesteps - 1, )) 
z_t[0] = 10

for i in range(num_timesteps - 1):
    mu_bar[:,i+1] = np.matmul(A_t,mu_t[:,i]) + B_t*u_t #prediction based estimate of state 
    cov_bar[i+1,:,:] = np.matmul(np.matmul(A_t,cov_t[i,:,:]),np.transpose(A_t)) + R_t #sigma_t
    K_t = np.matmul(np.matmul(cov_bar[i+1,:,:],np.transpose(C_t)), inv(np.matmul(C_t,np.matmul(cov_bar[i+1,:,:],np.transpose(C_t)))+Q_t))
    mu_t[:,i+1] = mu_bar[:,i+1] + np.matmul(K_t, z_t[i] - np.matmul(C_t,mu_bar[:,i+1])) #estimate of state taken into account the measurement
    cov_t[i+1,:,:] = np.matmul((np.eye(2) - np.matmul(K_t, C_t)), cov_bar[i+1,:,:])

#print(mu_t)
#print(mu_bar[:,1:]) #all rows, first column onwards
print(cov_t)
#print(cov_bar[1:,:,:]) #first matrix onwards, with all rows and columns
print(K_t)


