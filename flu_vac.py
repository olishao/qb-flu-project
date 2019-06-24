# FLU VACCINATION PROJECT CODE
# SUMMER 2019
# OLIVIA SHAO

import numpy as np
import matplotlib.pyplot as plt

WANING_RATE = 0.024

def immunity(t, t_vac=0):
    '''
    gives distribution for immunity of flu vaccine 
    inputs:
        t (array of integers): represents days
        t_vac (int): day of flu vaccination
    '''
    dist = np.exp(-WANING_RATE*t)
    unvac_days = np.array([0]*t_vac)
    dist = np.concatenate([unvac_days, dist])
    dist = dist[0:len(t)]
    return dist

def flu_foi(t):
    '''
    gives distribution for force of infection of flu
    input:
        t (array of integers): represents days
    '''
    a = 1/2
    k = 1/30
    p = np.pi
    b = 0.5
    return a*np.cos(k*t + p) + b

days_tot = np.arange(0, 180)
pct_flu_red = []
for day in days_tot:
    t_vac = day
    imm = immunity(days_tot, t_vac)
    foi = flu_foi(days_tot)
    pct_flu_red.append(sum(imm*foi))

plt.plot(days_tot, pct_flu_red)
plt.show()


immunity(np.arange(0, 180))