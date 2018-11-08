import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from importlib import reload
from sympy import *
from scipy.optimize import fsolve, minimize_scalar
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
E = metric.EllisWormhole(1, [t, r, th, ph], timelike=False, lambdify_modules='numpy')

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

anomaly = 3.728
args = (30, anomaly)
vals = {r0:30, th:anomaly}

impact_param = fsolve(Ff2, S.unstable_orbits, args, factor=0.1)[1]
bounds = [(30,30e+4),(impact_param,30)]

def impact_func(r, rO, theta):
    return (Rf2(r)-Rf2(rO)*np.sin(theta)**2) / (Sf2(r)*Rf2(r))

def test_lens(ths, rO=30, rS=None):
    if not rS:
        rS = 1e+4*rO
    rays = []
    for th in ths:
        rays.append(Ray([rO,0,0],[np.pi/2,th]))
    rays = lensing.static_spherical_lens(rays,rS,S)
    return np.array([ray.angles[1] for ray in rays])

def test_bv_lens(ths, rO=30, rS=None, general=False):
    if not rS:
        rS = rO/.77
    rays = []
    for th in ths:
        rays.append(Ray([rO,0,0],[np.pi/2,th]))
    if not general:
        rays = lensing.barriola_vilenkin_lens(rays,rS,B)
    else:
        rays = lensing.static_spherical_lens(rays,rS,B)
    return np.array([ray.angles[1] for ray in rays])

def test_ew_lens(ths, rO=30, rS=None):
    if not rS:
        rS = 1e+4*rO
    rays = []
    for th in ths:
        rays.append(Ray([rO,0,0],[np.pi/2,th]))
    rays = lensing.ellis_wormhole_lens(rays,rS,E,orient=np.sign(rO))
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

s = np.linspace(0,2*np.pi,1000)
#s = np.linspace(-np.pi,np.pi,1000)
rays = []
for val in np.linspace(0,2*np.pi,20):
    rays.append(Ray([300,0,0],[np.pi/2,val]))

def delta1(rO):
    inf = minimize_scalar(lambda r: np.sqrt(Rf2(r)/Rf2(rO)), method='bounded', bounds=(rO, 1e+4*rO))
    if inf['fun'] <= 1:
        return np.arcsin(inf['fun'])
    else:
        return np.pi/2
def delta2(rO):
    inf = minimize_scalar(lambda r: np.sqrt(Rf2(r)/Rf2(rO)), method='bounded', bounds=(0, rO))
    if inf['fun'] <= 1:
        return np.arcsin(inf['fun'])
    else:
        return np.pi/2

init_printing(num_columns=150)
