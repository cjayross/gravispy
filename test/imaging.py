import numpy as np
import itertools as it
from importlib import reload
from PIL import Image
from sympy import symbols
import gravispy.model.imagetransform as model
import gravispy.geom as geom
metric = geom.metric
lensing = geom.lensing
t,r,th,ph,M = symbols('t r theta phi M', positive=True)
S = metric.Schwarzschild(1, [t,r,th,ph], timelike=False, lambdify_modules='numpy')

#args = (3e+3,3e+7, S)
args = (3e+3,3e+7)
img = Image.open('earth.jpg')
#lens_map = model.generate_lens_map(lensing.schwarzschild_lens, img.size, args)
print('Generating lens map')
lens_map = model.generate_lens_map(lensing.trivial_lens, img.size, args)
print('Applying lens to image')
model.apply_lensing(img, lens_map)
#npz = np.load('lens_map.npz')
#keys = npz['keys']
#vals = npz['vals']
#k1,k2 = keys.T
#v1,v2 = vals.T
#lens_map = dict(zip(zip(k1,k2),zip(v1,v2)))
#tmp = tuple(map(int,lens_map[tuple(keys[-1])]))
#tmp = lens_map[1,0]
