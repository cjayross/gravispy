from importlib import reload
import numpy as np
import itertools as it
import gravispy.geom as geom
from sympy import symbols
lensing = geom.lensing
metric = geom.metric
unwrap = geom.unwrap
wrap = geom.wrap
pi = np.pi

t,r,th,ph = symbols('t r theta phi', real=True)
S = metric.Schwarzschild(1.,[t,r,th,ph],timelike=False)
args = (10,1e+5,S)

lin1 = np.linspace(0,2*pi)
lin2 = np.linspace(-pi,pi)
lin3 = np.linspace(-pi/2,pi/2)
res = [50,50]
x,y = map(np.arange,res)
x,y = map(np.ndarray.astype, [x,y], [int,int])
th,ph = geom.pix2sph(x,y,res)
k = {'th':np.abs(unwrap(th))>pi/2, 'ph':np.abs(unwrap(ph))>pi/2}

Ph = np.fromiter(lensing.static_spherical_lens(lin1,*args), np.float64)
