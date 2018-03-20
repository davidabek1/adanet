'''
Generate "two spirals" dataset with N instances.
degrees controls the length of the spirals
start determines how far from the origin the spirals start, in degrees
noise displaces the instances from the spiral. 
 0 is no noise, at 1 the spirals will start overlapping
'''

import numpy as np
import math

def twospirals(*args):  #N, degrees, start, noise

    if len(args) < 1:
        N = 2000
    else:
        N = args[0]
    if len(args) < 2:
        degrees = 570
    else:
        degrees = args[1]
    if len(args) < 3:
        start = 90
    else:
        start = args[2]
    if len(args) < 5:
        noise = 0.2
    else:
        noise = args[3]
    
    deg2rad = (2*math.pi)/360
    start = start * deg2rad

    N1 = math.floor(N/2)
    N2 = N-N1
    
    n = start + np.dot(np.dot(np.sqrt(np.random.random((N1,1))),degrees),deg2rad)
    d1 = np.hstack([-np.cos(n)*n + np.dot(np.random.random((N1,1)),noise),np.sin(n)*n+np.dot(np.random.random((N1,1)),noise),np.zeros((N1,1))])
    
    n = start + np.dot(np.dot(np.sqrt(np.random.random((N1,1))),degrees),deg2rad)   
    d2 = np.hstack([np.cos(n)*n + np.dot(np.random.random((N2,1)),noise),-np.sin(n)*n+np.dot(np.random.random((N2,1)),noise),np.ones((N2,1),float)])
    
    data = np.vstack((d1,d2))
    return data