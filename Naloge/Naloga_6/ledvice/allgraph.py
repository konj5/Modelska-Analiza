import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re

models = ["$N_0 \cdot e^{-L_0t}$", "$N_0 \cdot e^{-L_0t} + N_1 \cdot e^{-L_1t}$", "$N_0 \cdot e^{-L_0t} +C_0$", "$N_0 \cdot e^{-L_0t} + N_1 \cdot e^{-L_1t} + C_0$", "$N_0 \cdot e^{-L_0\sqrt{t}} $", "$N_0 \cdot e^{-L_0\sqrt{t}} + N_1 \cdot e^{-L_1\sqrt{t}}$", "$N_0 \cdot e^{-L_0\sqrt{t}} + C_0 $", "$N_0 \cdot e^{-L_0\sqrt{t}} + N_1 \cdot e^{-L_1\sqrt{t}} + C_0$"]
n_parameters = [2,4,3,5,2,4,3,5]
chis = [26758279.60, 442932.23, 3022383.29, 208155.26, 1818560.91, 1343047.73, 1644405.34, 863786.33]



import seaborn as sns
palette = sns.color_palette(None, 8)


plt.scatter(n_parameters, chis, c=palette)
plt.xlabel("Število parametrov")
plt.ylabel("$\\chi^2$")
plt.yscale("log")
plt.show()

