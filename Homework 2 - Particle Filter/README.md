# HW2 Particle Filter

##### Aparna Penmetcha

Run the python program PRHW2_ParticeFilter which utilizes numpy and matplotlib. 
Three images (MarioMap, BayMap and CityMap) were given and all can be used with this program to localize a drone over each map. 

For the best results of the capabilities of the program, BayMap should be used, with the image size (m) = 200 and number of particles (N) = 200.

Once the program is initiated, a window with the whole image will show up. There will be a black circle (the drone) and several particles (red) on the map. As you go through the timesteps by pressing the enter key in the terminal, the window will update each time with a new position of the drone and particles. 

The simulation has a maximum number of 100 timesteps and will end the program if the particles do not converge before 100.