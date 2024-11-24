import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import cm, colors

def fun(t,v, args):
    vdot = np.zeros_like(v)
    x, y, z, u, v = v
    a, b, c, d, e = args


    vdot[0] = -a*x + b * u**2 - e*x*v
    vdot[1] = d*z*v - c*u*y
    vdot[2] = c*u*y - d*z*v + e*v*x
    vdot[3] = a*x - b*u**2 - c*u*y + d*z*v + e*v*x
    vdot[4] = c*u*y - d*z*v - e*v*x
    return vdot


def solve(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/20000)
    return sol.t, sol.y


beta = 1

a=1
b=100
c=5
d=100
e=250




ts,vs = solve([1/(beta+1),beta/(beta+1),0,0,0], fun,[a,b,c,d,e],0,100)


##############
def fun(t,v, args):
    x, y, z = v
    vdot = np.zeros_like(v)
    k, m, = args

    zdot = k * np.sqrt(x) * y / (m + z/x)

    vdot[0] = -zdot/2
    vdot[1] = -zdot/2
    vdot[2] = zdot
    return vdot


def solve(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/20000)
    return sol.t, sol.y


beta = 1

m = 2.5

k = 2.5


ks = np.linspace(0,4,100)

ts0,vs0 = solve([1/(beta+1),beta/(beta+1),0], fun,[k,m],0,100)

plt.plot(ts, np.abs(vs0[0,:]), label = "[Br$_2$]", c = "red", linestyle = "dashed")
plt.plot(ts, np.abs(vs0[1,:]), label = "[H$_2$]", c = "blue", linestyle = "dashed")
plt.plot(ts, np.abs(vs0[2,:]), label = "[HBr]", c = "green", linestyle = "dashed")


plt.plot(ts, np.abs(vs[0,:]), label = "[Br$_2$]", c = "red")
plt.plot(ts, np.abs(vs[1,:]), label = "[H$_2$]", c = "blue")
plt.plot(ts, np.abs(vs[2,:]), label = "[HBr]", c = "green")
plt.legend()
plt.show()

#######

plt.figure(figsize=(4,3))

plt.plot(ts, np.abs(vs0[0,:] - vs[0,:]), label = "[Br$_2$]")
plt.plot(ts, np.abs(vs0[1,:] - vs[1,:]), label = "[H$_2$]")
plt.plot(ts, np.abs(vs0[2,:] - vs[2,:]), label = "[HBr]")



plt.ylabel("Absolutna napaka")
plt.xlabel("t")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(4,3))

plt.plot(ts, np.abs(vs0[0,:] - vs[0,:]), label = "[Br$_2$]")
plt.plot(ts, np.abs(vs0[1,:] - vs[1,:]), label = "[H$_2$]")
plt.plot(ts, np.abs(vs0[2,:] - vs[2,:]), label = "[HBr]")


plt.ylabel("Absolutna napaka")
plt.xlabel("t")
plt.yscale("log")
plt.tight_layout()
plt.legend()
plt.show()