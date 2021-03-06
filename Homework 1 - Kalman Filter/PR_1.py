# Kalman Filters Homework 1 
# Aparna Penmetcha 
# Problem 1.3/1.4 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from math import log, atan2, pi

A_t = [[1, 1],
      [0, 1]]

A_t = np.array(A_t)

B_t = 0
u_t = 0

sigma_w = 1 

R = sigma_w**2*(np.matmul(np.array([[0.5],[1]]),np.array([[0.5,1]])))
#epsilon = np.random.multivariate_normal(np.array([0, 0]) ,R)

num_timesteps = 6
mu_t = np.zeros((2,num_timesteps))
cov = np.zeros((num_timesteps,2,2))
#m_states = np.zeros((2,num_timesteps))

for i in range(num_timesteps - 1):

    mu_t[:,i+1] = np.matmul(A_t,mu_t[:,i]) + B_t*u_t
    cov[i+1,:,:] = np.matmul(np.matmul(A_t,cov[i,:,:]),np.transpose(A_t))+R
    #m_states[:,i+1] = np.matmul(A_t, m_states[:,i]) + B_t*u_t + epsilon.T


print('======== mu_t ========')
print(mu_t)
print('======== cov_t ========')
print(cov)
#print('======== M_states ========')
#print(m_states)

plt.scatter(mu_t[0,:],mu_t[1,:])
plt.suptitle('Joint Posterior Over Position and Velocity' )
plt.xlabel('position (x)')
plt.ylabel('velocity (x_dot)')
# plt.show()


def generateEllipseFunction(covMatrix):
    majorAxis = None
    minorAxis = None
    angle = None

    sigma_x = covMatrix[0,0]
    sigma_y = covMatrix[1,1]

    s = -2 * log(1 - 0.68) #Standard deviation = 1, confidence interval 68%
    majorAxis = 2 * sigma_x * s 
    minorAxis = 2 * sigma_y * s

    eigenvalues, eigenvectors = np.linalg.eig(covMatrix)

    largestEigenValue = None
    largestEigenIndex = None
    smallestEigenValue = None
    smallestEigenIndex = None

    print('eigen at  t') 
    print(eigenvalues)
    print(eigenvectors)

    if eigenvalues[1] > eigenvalues[0]:
        largestEigenValue = eigenvalues[1]
        largestEigenIndex = 1
        smallestEigenValue = eigenvalues[0]
        smallestEigenIndex = 0
    else:
        smallestEigenValue = eigenvalues[1]
        smallestEigenIndex = 1
        largestEigenValue = eigenvalues[0]
        largestEigenIndex = 0

    largestEigenVector = eigenvectors[:, largestEigenIndex]
    smallestEigenVector = eigenvectors[:, smallestEigenIndex]

    angle = atan2(largestEigenVector[1], largestEigenVector[0])
    # add 2pi if angle is negative
    if angle < 0:
        angle = angle + 2 * pi
    # convert from radians to degrees
    angle = angle * 180 / pi


    return [majorAxis, minorAxis, angle]

ellipses = []
for c in cov:
    ellipseFunction = generateEllipseFunction(c)
    ellipses.append(ellipseFunction)

def generateEllipseFunction(covMatrix):
    majorAxis = None
    minorAxis = None
    angle = None

    sigma_x = covMatrix[0,0]
    sigma_y = covMatrix[1,1]

    s = -2 * log(1 - 0.68)

    eigenvalues, eigenvectors = np.linalg.eig(covMatrix)

    largestEigenValue = None
    largestEigenIndex = None
    smallestEigenValue = None
    smallestEigenIndex = None

    if eigenvalues[1] > eigenvalues[0]:
        largestEigenValue = eigenvalues[1]
        largestEigenIndex = 1
        smallestEigenValue = eigenvalues[0]
        smallestEigenIndex = 0
    else:
        smallestEigenValue = eigenvalues[1]
        smallestEigenIndex = 1
        largestEigenValue = eigenvalues[0]
        largestEigenIndex = 0

    largestEigenVector = eigenvectors[:, largestEigenIndex]
    smallestEigenVector = eigenvectors[:, smallestEigenIndex]



    majorAxis = 2 * (largestEigenValue * s) ** 0.5
    minorAxis = 2 * (smallestEigenValue * s) ** 0.5
    angle = atan2(largestEigenVector[1], largestEigenVector[0])
    # add 2pi if angle is negative
    if angle < 0:
        angle = angle + 2 * pi
    # convert from radians to degrees
    angle = angle * 180 / pi


    return [majorAxis, minorAxis, angle]

ellipses = []
for c in cov:
    ellipseFunction = generateEllipseFunction(c)
    ellipses.append(ellipseFunction)


### Tung's Formatting ###

fig, axs1 = plt.subplots(1, 3, figsize=(10, 4))
for index, ellipse in enumerate(ellipses[:3]):
    ax = axs1[index]
    ax.add_patch(Ellipse(
        (0, 0),
        width=ellipse[0],
        height=ellipse[1],
        angle=ellipse[2],
        fill=True,
        edgecolor = 'black')
    )
    ax.axvline(c='grey', lw=1)
    ax.axhline(c='grey', lw=1)
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_title("Uncertainty Ellipse at t = %s" % index)

fig, axs2 = plt.subplots(1, 3, figsize=(10,4 ))
for index, ellipse in enumerate(ellipses[3:]):
    ax = axs2[index]
    ax.add_patch(Ellipse(
        (0, 0),
        width=ellipse[0],
        height=ellipse[1],
        angle=ellipse[2],
        fill=True,
        edgecolor = 'black')
    )
    ax.axvline(c='grey', lw=1)
    ax.axhline(c='grey', lw=1)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_title(" Uncertainty Ellipse at t = %s"  % (index+3))

plt.show()
