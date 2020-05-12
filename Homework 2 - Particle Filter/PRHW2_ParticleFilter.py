# Probabilistic Robotics Homework 2 
# Particle Filter 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np 
import random 
import math 

def get_img(pos,img,m,column,row):
    return img[max(0,math.floor(pos[0] - (m/2))): min(column,math.floor(pos[0] + (m/2))),max(0,math.floor(pos[1] - (m/2))): min(row,math.floor(pos[1] + (m/2))),:]
        

def plot_measurement(pos,img,m,column,row):
    # makes sure to give square image, and that the image taken is on map and not off the map 
        new_image = get_img(pos,img,m,column,row)
        plt.figure()
        imgplot = plt.imshow(new_image)
        plt.show()
        pass

def error_meas(z,reference_img,m):
    # z observed image 
    # true_pos - current position 
    temp = np.zeros((m,m,img.shape[2]))
    #print(temp.shape)
    #print(reference_img.shape)
    temp[0:reference_img.shape[0],0:reference_img.shape[1],:] = reference_img

    temp1 = np.zeros((m,m,img.shape[2]))
    temp1[0:z.shape[0],0:z.shape[1],:] = z
    dif = math.sqrt(np.sum(np.power(temp - temp1,2)))
    return dif

# loading and importing PNG image in 
img = mpimg.imread('C:/Users/aparn/Desktop/PR Homework 2/BayMap.png')
#print(img)
#print(type(img))

#rows, columns 
#print(img.shape) 

#size of the whole map(image) 
row,column = img.shape[:2]

# initial random positions 
y_initial = random.randrange(0, row)
x_initial = random.randrange(0, column)

# state vector --> x and y position of drone 
s_vector = np.transpose([x_initial,y_initial])

num_timesteps = 50

#Initialize particle Filter
N = 100 #number of particles
particles = np.array([[ random.randrange(0, column) for i in range(N)],[ random.randrange(0, row) for i in range(N)]])
weights = np.ones((N,1))
# m value
m = 100
sigma_p = min(column,row)/50

########### difference between reference and observation image? Part 1 Simulation bullet point 3 ############

converging = True
t = 0
#for i in range(num_timesteps): 
while converging and t < 100:
    t = t+1    
# observation image 
    true_pos = s_vector

# plot original state 
    #plot_measurement(s_vector, img) # Observation image
    z = get_img(s_vector,img,m,column,row)

    #plot the map, true position and the particles 

    plt.figure()
    ax = plt.gca()
    
    #plt.show(block=False)
    #input("Press Enter to continue - check clearing figure...")
    imgplot = ax.imshow(img)
    ax.add_image(imgplot)

    #drone on map 
    c = plt.Circle((s_vector),30,color = 'black')
    ax.add_artist(c)
    c = plt.Circle((s_vector),10,color = 'pink')
    ax.add_artist(c)
    #dots on map (particles)
    if t > 1:
        for j in range(N):
            c = plt.Circle((particles[:,j]),weights[j]*1000,color = 'r')
            ax.add_artist(c)

    #no need to close window anymore
    plt.show(block=False)

    
    errors = []
    for j in range(N):
        p_coordinates = particles[:,j]
        reference_img = get_img(p_coordinates,img,m,column,row)
        #print(p_coordinates)
        errors.append(error_meas(z,reference_img,m))
    
    #find weights accoring to the error values
    #adjust errors to not have "0" error
    errors = np.array(errors)+0.1
    weights = np.reciprocal(errors)/sum(np.reciprocal(errors))
    #print(errors)
    print(weights)
    #print(particles)

    #resample particles accorind to weights
    index = range(N)
    resampled_index = np.random.choice(index,size = len(index),replace=True, p=weights)

    particles_resampled = particles[:,resampled_index]

    #distance 1 = 100pixles 
    dist = 50
    
    # Find the random movement vector and update the state
    found_new = False
    while not found_new :

        dir = random.uniform(0, 2*math.pi)
        dx = math.floor(math.cos(dir)*dist)
        dy = math.floor(math.sin(dir)*dist)

        rand_movement_vector = np.transpose([dx,dy])

        try_state = s_vector + rand_movement_vector
        # conditions for being outside of map (if any of these conditions are true; then you are outside of map)
        if not (try_state[0]<0 or try_state[0] > column or try_state[1]<0 or try_state[1] > row):
            found_new = True


    #move all particles according to the rand_movement_vector
    particles_resampled = particles_resampled + rand_movement_vector.reshape(2,1)
    for j in range(particles_resampled.shape[1]):
        print(particles_resampled[:,j])
        print(np.random.normal(0,dist/2,2).reshape(2,1))
        particles_resampled[:,j] = particles_resampled[:,j] + np.random.normal(0,sigma_p,2)
    for j,x in enumerate(particles_resampled[0,:]):
        if x < 0:
            particles_resampled[0,j] = 0
        elif x > column:
            particles_resampled[0,j] = column
    for j,y in enumerate(particles_resampled[1,:]):
        if y < 0:
            particles_resampled[1,j] = 0
        elif y > row:
            particles_resampled[1,j] = row

    #update particle variable
    particles = None
    particles = particles_resampled

    #check if each particle is within the  observation image
    count = 0
    for j in range(N):
        p_coord = np.floor(particles[:,j])
        if ((p_coord[0] > s_vector[0] - 3*m/4) and (p_coord[0] < s_vector[0] + 3*m/4) and
                (p_coord[1] > s_vector[1] - 3*m/4) and (p_coord[1] < s_vector[1] + 3*m/4)):
                count = count+1
    if count/N > 0.7:
        converging = False



    #plot of next image 
    #plot_measurement(try_state, img) 

    # higher sigma, higher noise 
    sigma = 5
    s_vector = np.floor(try_state + np.transpose([random.gauss(0,sigma),random.gauss(0,sigma)]))

    input("Press Enter to continue...")
    plt.close(1)
if t == 100:
    print('Did not converege')
else:
    print('{} timesteps used to obtain 70% of the points within the true position of the drone'.format(t))
input("SAVE PICTURES and press enter to exit")
