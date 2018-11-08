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
s = np.linspace(0,2*np.pi,1000)

S = metric.Schwarzschild(1, [t, r, th, ph], timelike=False, lambdify_modules='numpy')
B = metric.BarriolaVilenkin(1/3.7, [t, r, th, ph], timelike=False, lambdify_modules='numpy')
E = metric.EllisWormhole(1, [t, r, th, ph], timelike=False, lambdify_modules='numpy')

Af2 = S.conformal_factor(generator=True)
Sf2 = S.radial_factor(generator=True)
Rf2 = S.angular_factor(generator=True)

P = lambda r,r0,th:\
        np.sqrt(Rf2(r0)*Sf2(r)
                / (Rf2(r)*(Rf2(r)-Rf2(r0)*np.sin(th)**2)))\
        * np.sin(th)

def impact_func(r, rO, theta):
    return (Rf2(r)-Rf2(rO)*np.sin(theta)**2) / (Sf2(r)*Rf2(r))

def test_lens(ths, rO=30, rS=None):
    if not rS:
        rS = 1e+4*rO
    return lensing.static_spherical_lens(ths, rO, rS, S)

def test_bv_lens(ths, rO=30, rS=None, general=False):
    if not rS:
        rS = rO/.77
    if not general:
        return lensing.barriola_vilenkin_lens(ths,rO,rS,B)
    else:
        return lensing.static_spherical_lens(ths,rO,rS,B)

def test_ew_lens(ths, rO=30, rS=None):
    if not rS:
        rS = 1e+4*rO
    return lensing.ellis_wormhole_lens(ths,rO,rS,E)

def test_thin_lens(ths, rO=30, rS=None):
    if not rS:
        rS = 1e+4*rO
    return lensing.schwarzschild_thin_lens(ths,rO,rS,S)

def test_trivial_lens(ths, rO=30, rS=None):
    if not rS:
        rS = 1e+4*rO
    return lensing.trivial_lens(ths, rO, rS)

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
