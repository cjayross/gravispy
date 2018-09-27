from time import time
t0 = time()
def time_stamp (message):
    print(message + ': %f' % (time() - t0))

import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.patches import Circle
from scipy.integrate import solve_ivp
from gravispy.geom.metric import *
from gravispy.ext.gravipy import *
time_stamp('Loaded modules')

t,r,theta,phi,M,m,E,L = symbols('t r theta phi M m E L', real=True)
F = Function('F')
G = Function('G')
chi = Coordinates('chi', [t,r,theta,phi])

g = Schwarzschild('g', chi, M)
time_stamp('Loaded metric tensor')

g.set_principal_function(F, G, m, E, L)
geodesics = g.geodesic_equations()
time_stamp('Defined Geodesic equations')
HJ = g.hamilton_jacobi()
time_stamp('Defined Hamilton-Jacobi equation')

class Vals (object):
    def __init__ (self):
        self.vals = { E : 0.4429, L : 23, m : 0, M : 10, 'root' : 1, diff(G(theta), theta) : 0, theta : pi / 2 }
        self.dF = solve(HJ, diff(F(r), r))[self.vals['root']].subs(self.vals)
        self.vals.update({ diff(F(r), r) : self.dF })
        self.dR = (geodesics[1]/geodesics[0]).subs(self.vals)
        self.dPhi = (geodesics[3]/geodesics[0]).subs(self.vals)
        self.ldR = lambdify(r, self.dR, 'numpy')
        self.ldPhi = lambdify(r, self.dPhi, 'numpy')
        self.ld2R = lambdify(r, diff(self.dR, r), 'numpy')
        self.ld2Phi = lambdify(r, diff(self.dPhi, r), 'numpy')

    def set_vals (self, **args):
        self.vals = { E : args.setdefault('E', self.vals[E]),\
                      L : args.setdefault('L', self.vals[L]),\
                      m : args.setdefault('m', self.vals[m]),
                      M : args.setdefault('M', self.vals[M]),\
                      'root' : args.setdefault('root', self.vals['root']),\
                      diff(G(theta), theta) : 0,\
                      theta : pi / 2 }
        self.dF = solve(HJ, diff(F(r), r))[args.setdefault('root', v.vals['root'])].subs(self.vals)
        self.vals.update({ diff(F(r), r) : self.dF })
        self.dR = (geodesics[1]/geodesics[0]).subs(self.vals)
        self.dPhi = (geodesics[3]/geodesics[0]).subs(self.vals)
        self.ldR = lambdify(r, self.dR, 'numpy')
        self.ldPhi = lambdify(r, self.dPhi, 'numpy')
        self.ld2R = lambdify(r, diff(self.dR, r), 'numpy')
        self.ld2Phi = lambdify(r, diff(self.dPhi, r), 'numpy')

v = Vals()
step = 1e+3
tt = np.linspace(0,1e+3,step)

def f(t, y): return np.array([ v.ldR(y[0]), v.ldPhi(y[0]) ])
def df(t, y): return np.array([[v.ld2R(y[0]), 0], [v.ld2Phi(y[0]), 0]])

# set this to the second root (we need to look into how the solution depends on this)
v.set_vals(root=1)

t1 = time()
sol = solve_ivp(f, [min(tt), max(tt)], [10*v.vals[M], 0.3], method='LSODA', jac=df, t_eval=tt)
print('Solved null geodesic in %f' % (time() - t1))
print('Number of function calls: %d' % sol.nfev)

xx = sol.y[0] * np.cos(sol.y[1])
yy = sol.y[0] * np.sin(sol.y[1])
fig, ax = plt.subplots()
plt.plot(xx,yy)
ax.add_patch(plt.Circle((0,0), 2*v.vals[M], fill=True, color='k'))
plt.show()
