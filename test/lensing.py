import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from importlib import reload
from sympy import *
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import solve_ivp, romberg, quad
import gravispy.geom as geom
from basic_units import radians
wrap = geom.wrap
unwrap = geom.unwrap
metric = geom.metric
lensing = geom.lensing
Ray = geom.Ray
Plane = geom.Plane
Sphere = geom.Sphere
t, r, th, ph, r0, l0, l_P, l, M = symbols('t r theta phi r0 l0 l_P l M', positive=True)
s = np.linspace(0,2*np.pi,1000)
args = (30, 3*np.pi/4)
rO, theta = args
l_R = 1/2/sqrt(2)

S = metric.Schwarzschild(1, [t, r, th, ph], timelike=False, lambdify_modules='numpy')
SM = metric.Schwarzschild(M, [t, r, th, ph], timelike=False, lambdify_modules='numpy')
B = metric.BarriolaVilenkin(1/3.7, [t, r, th, ph], timelike=False, lambdify_modules='numpy')
E = metric.EllisWormhole(1, [t, r, th, ph], timelike=False, lambdify_modules='numpy')

Sf2 = S.radial_factor(generator=True)
Rf2 = S.angular_factor(generator=True)
S2 = SM.radial_factor()
R2 = SM.angular_factor()
P = sqrt(R2.subs({r:r0})*S2 / (R2*(R2-R2.subs({r:r0})*sin(th)**2))) * sin(th)
Pf = lambdify(r, P.subs({r0:30, th:theta, M:1}))

def T_impact_func(r, rO, theta):
    return (Rf2(r)-Rf2(rO)*np.sin(theta)**2) / (Sf2(r)*Rf2(r))

r_P = brentq(T_impact_func, 3, rO, args=args)
l_P = 1/np.sqrt(2)/r_P
q1 = 2*l_P*(1-2*np.sqrt(2)*l_P)
q2 = 1-6*np.sqrt(2)*l_P
q3 = 2*np.sqrt(2)
T_P = lambda q: np.sqrt(2)*(l_P-q)**2/np.sqrt(q1*q-q2*q**2-q3*q**3)

def T_Pf(r):
    l = 1/np.sqrt(2)/r
    return T_P(l_P-l)

def T_phi_func(rval, rO, theta):
    num = Rf2(rO)*Sf2(rval)
    den = Rf2(rval) * (Rf2(rval)-Rf2(rO)*np.sin(theta)**2)
    return np.sin(theta) * np.sqrt(num/den)

def T_lens(ths, rO=30, rS=None):
    if not rS:
        rS = 1e+4*rO
    return np.fromiter(lensing.static_spherical_lens(ths, rO, rS, S), np.float32)

def T_sc_lens(ths, rO=30, rS=None):
    if not rS:
        rS = 1e+4*rO
    return np.fromiter(lensing.schwarzschild_lens(ths, rO, rS, S), np.float32)

def T_bv_lens(ths, rO=30, rS=None, general=False):
    if not rS:
        rS = rO/.77
    if not general:
        return np.fromiter(lensing.barriola_vilenkin_lens(ths,rO,rS,B), np.float32)
    else:
        return np.fromiter(lensing.static_spherical_lens(ths,rO,rS,B), np.float32)

def T_ew_lens(ths, rO=30, rS=None):
    if not rS:
        rS = 1e+4*rO
    return np.fromiter(lensing.ellis_wormhole_lens(ths,rO,rS,E), np.float32)

def T_thin_lens(ths, rO=30, rS=None):
    if not rS:
        rS = 1e+4*rO
    return np.fromiter(lensing.schwarzschild_thin_lens(ths,rO,rS,S), np.float32)

def T_trivial_lens(ths, rO=30, rS=None):
    if not rS:
        rS = 1e+4*rO
    return np.fromiter(lensing.trivial_lens(ths, rO, rS), np.float32)

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
testS1 = T_sc_lens(unwrap(s),10)
testS2 = T_lens(unwrap(s),10)
testS3 = T_thin_lens(unwrap(s),10)
plt.xlabel(r'Observation Angle, $\Theta$')
plt.ylabel(r'Source Angular Position, $\Phi$')
plt.plot(s*radians,unwrap(testS1+np.pi),label='Explicit Lens')
plt.plot(s*radians,unwrap(testS2+np.pi),alpha=.7,label='General Lens')
plt.plot(s*radians,unwrap(testS3+np.pi),'--k',alpha=.7,label='Thin Lens')
plt.xlim(0*radians,2*(np.pi*radians))
plt.ylim(-np.pi*radians,np.pi*radians)
plt.title('Schwarzschild Lensing Methods')
plt.legend()
plt.savefig('sc_lensing.png')
