from time import time
t0 = time()
def time_stamp (message):
    print(message + ': %f' % (time() - t0))

from gravispy.geom.metric import *
from gravispy.ext.gravipy import *
time_stamp('Loaded modules')

t,r,theta,phi,M,m,E,L = symbols('t r theta phi M m E L', real=True)
F = Function('F')
G = Function('G')
chi = Coordinates('chi', [t,r,theta,phi])

g = Schwarzschild('g', chi, M)
time_stamp('Loaded metric tensor')

g.set_principal_function(F, G, [m, E, L])
geodesics = g.geodesic_functions()
time_stamp('Defined Geodesic equations')
HJ = g.hamilton_jacobi()
time_stamp('Defined Hamilton-Jacobi equation')
