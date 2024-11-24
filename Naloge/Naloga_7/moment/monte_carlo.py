import numpy as np
import time

rng = np.random.default_rng(time.time_ns())

def monte_carlo(f:callable, limits:list, constraint:callable, N:int):
    """
    f - finction to integrate, takes Ndimx1 arguments
    limits - area that includes at least the whole area, where the constrain is non-zero Ndim list of type [xmin, xmax]
    constraint - function that returns 0 if outside the integration area and 1 
    N - number of random points
    """
    Ndim = len(limits)

    def get_random_guess():
        ret = np.zeros(Ndim)
        for i in range(Ndim):
            ret[i] = limits[i][0] + (limits[i][1] - limits[i][0]) * np.random.random()
        return ret
    
    def get_V0():
        ret = 1
        for i in range(Ndim):
            ret *= limits[i][1] - limits[i][0]
        return ret
    
    val = 0
    for _ in range(N):
        x = get_random_guess()
        val += f(x) * constraint(x)

    return val/N * get_V0()

def monte_carlo_r(f:callable, limits:list, constraint:callable, N:int):
    """
    f - finction to integrate, takes Ndimx1 arguments
    limits - area that includes at least the whole area, where the constrain is non-zero Ndim list of type [xmin, xmax]
    constraint - function that returns 0 if outside the integration area and 1 
    N - number of random points
    """
    Ndim = len(limits)

    def get_random_guess():
        return [rng.random()*(1/3), np.acos(2*np.random.random()-1), 2 * np.pi * np.random.random()]
    
    def get_V0():
        return 4*np.pi/3
    
    val = 0
    for _ in range(N):
        x = get_random_guess()
        val += f(x) * constraint(x)

    return val/N * get_V0()

#print(monte_carlo(lambda x:x[0], [[0,2]], lambda x:  1 if x < 1 else 0, 100000))



