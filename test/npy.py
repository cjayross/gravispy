from importlib import reload
import numpy as np
import itertools as it
import gravispy.geom as geom
unwrap = geom.unwrap
wrap = geom.wrap
pi = np.pi
lin1 = np.linspace(0,2*pi)
lin2 = np.linspace(-pi,pi)
lin3 = np.linspace(-pi/2,pi/2)
res = [50,50]
x,y = map(np.arange,res)
x,y = map(np.ndarray.astype, [x,y], [int,int])
th,ph = geom.pixel2sph(x,y,res)
k = {'th':np.abs(unwrap(th))>pi/2, 'ph':np.abs(unwrap(ph))>pi/2}

def asin(a):
    k = np.abs(unwrap(a))>pi/2
    return wrap(pi*k + (-1)**k*np.arcsin(np.sin(a)))

def acos(a):
    sign = np.sign(unwrap(a))
    return wrap(sign*np.arccos(np.cos(a)))
