import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
from sympy import *
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp, romberg, quad
import gravispy.geom as geom
metric = geom.metric
lensing = geom.lensing
Ray = geom.Ray
Plane = geom.Plane
Sphere = geom.Sphere
t, r, th, ph, r0 = symbols('t r theta phi r0', positive=True)
S = metric.Schwarzschild(1, [t, r, th, ph], timelike=False, lambdify_modules='numpy')
B = metric.BarriolaVilenkin(1/3.7, [t, r, th, ph], timelike=False, lambdify_modules='numpy')
Af2 = S.conformal_factor(generator=True)
Sf2 = S.radial_factor(generator=True)
Rf2 = S.angular_factor(generator=True)
A2 = S.conformal_factor()
S2 = S.radial_factor()
R2 = S.angular_factor()

F2 = (R2-R2.subs({r:r0})*sin(th)**2)/(R2*S2)
Ff2 = lambdify((r,r0,th),F2,'numpy')
F = sqrt(F2)
Ff = lambdify((r,r0,th),F,'numpy')

G2 = R2-R2.subs({r:r0})*sin(th)**2
Gf2 = lambdify((r,r0,th),G2,'numpy')
G = sqrt(G2)
Gf = lambdify((r,r0,th),G,'numpy')

P = lambda r,r0,th:\
        np.sqrt(Rf2(r0)*Sf2(r)
                / (Rf2(r)*(Rf2(r)-Rf2(r0)*np.sin(th)**2)))\
        * np.sin(th)

args = (10, np.pi/4)
vals = {r0:10, th:pi/4}

impact_param = fsolve(Ff2, S.unstable_orbits, args, factor=0.1)[1]

def test_lens(ths, rO=30):
    rays = []
    for th in ths:
        rays.append(Ray([rO,0,0],[np.pi/2,th]))
    rays = lensing.static_spherical_grav_lens(rays,1e+4*rO,S)

    return np.array([ray.angles[1] for ray in rays])

def test_thin_lens(ths, rO=30):
    rays = []
    for th in ths:
        rays.append(Ray([rO,0,0],[np.pi/2,th]))
    rays = lensing.schwarzschild_thin_lens(rays, S)
    points = [ray(geom.sphere_intersect(ray, Sphere([0,0,0],1e+4*rO))) for ray in rays]
    return np.array([np.arctan2(point[1], point[0]) for point in points])

def test_trivial_lens(ths, rO=30):
    rays = []
    for th in ths:
        rays.append(Ray([rO,0,0],[np.pi/2,th]))
    points = [ray(geom.sphere_intersect(ray, Sphere([0,0,0],1e+4*rO))) for ray in rays]
    return np.array([np.arctan2(point[1], point[0]) for point in points])

s = np.linspace(-2,2,1000)
t = np.pi - s

test_trivial = test_trivial_lens(t, 300)
test_thin = test_thin_lens(t, 300)
#test = test_lens(t, 300)

test_trivial = [phi-np.pi if phi > 0 else phi+np.pi for phi in test_trivial]
test_thin = [phi-np.pi if phi > 0 else phi+np.pi for phi in test_thin]

init_printing(num_columns=150)
