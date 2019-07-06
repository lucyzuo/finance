from time import time
from math import exp, sqrt, log
from random import gauss, seed

# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

# variables
S0 = 100
T = 1.0
r = 0.05
sigma = 0.20
N = 50
dt = T/N
I = 5000

seed(2000) # makes the same numbers appear each time; the parameter is arbitrary
S=[] # container array
for i in range(I): # I = 5000
    path = []
    for t in range(N+1): # N = 50
        if t == 0:
            path.append(S0)
        else:
            wt = gauss(0.0,1.0) # Gaussian distribution with mean 0 and standard deviation 1
            # draw a random number from normal distribution
            # add code for St 
            St =path[t-1] * exp((r-0.5*sigma**2) * dt + sigma *sqrt(dt) *wt)
            path.append(St)
    S.append(path)
    
plt.plot(S[1])
plt.plot(S[2])
plt.plot(S[3])
plt.plot(S[4])
plt.plot(S[5])
plt.grid(True)
plt.xlabel('time step')
plt.ylabel('idenx label')