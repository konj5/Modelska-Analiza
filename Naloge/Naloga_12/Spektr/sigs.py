import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re, sys
np.set_printoptions(linewidth=10000, threshold=sys.maxsize)

import spectrum
from nitime import algorithms as alg

data2, data3, data_co2 = [], [], []

with open("Naloge\\Naloga_12\\val2.dat", mode = "r") as f:
    lines = f.readlines()

    for line in lines:
        data2.append(float(line))

with open("Naloge\\Naloga_12\\val3.dat", mode = "r") as f:
    lines = f.readlines()

    for line in lines:
        data3.append(float(line))


#*******************************************************************
#*** Atmospheric CO2 concentrations (ppmv) derived from in situ  ***
#*** air samples collected at Mauna Loa Observatory, Hawaii      ***
#***                                                             ***
#*** Source: C.D. Keeling                                        ***
#***         T.P. Whorf, and the Carbon Dioxide Research Group   ***
#***         Scripps Institution of Oceanography (SIO)           ***
#***         University of California                            ***
#***         La Jolla, California USA 92093-0444                 ***
#***                                                             ***
#*** May 2005                                                    ***
#***                                                             ***
#*******************************************************************
#Monthly values are expressed in parts per million (ppm) and reported in the 2003A SIO manometric mole 
#fraction scale.  The monthly values have been adjusted to the 15th of each month.  Missing values are 
#denoted by -99.99. 

with open("Naloge\\Naloga_12\\co2.dat", mode = "r") as f:
    lines = f.readlines()

    for line in lines:
        data_co2.append([float(line.split(" ")[0]), float(line.split(" ")[1]) if float(line.split(" ")[1]) != -99.99 else data_co2[-1][1]])

data2 = np.array(data2)
data3 = np.array(data3)     
data_co2 = np.array(data_co2)


fig, axs = plt.subplots(1,3)
ax1, ax2, ax3 = axs

ax1.plot(data2)
ax2.plot(data3)
ax3.plot(data_co2[:,1])

ax1.set_title("val2.dat")
ax2.set_title("val3.dat")
ax3.set_title("co2.dat")

plt.show()


