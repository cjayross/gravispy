from metric import *
from gravipy import *
t,r,theta,phi,M,m,E,L = symbols('t r theta phi M m E L', real=True)
F = Function('F')
G = Function('G')
chi = Coordinates('chi', [t,r,theta,phi])

g = Schwarzschild('g', chi, M)
g.set_principal_function(F, G, [m, E, L])
geodesics = g.geodesic_functions()
HJ = g.hamilton_jacobi()
