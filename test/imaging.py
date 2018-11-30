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

#np.seterr(all='raise')

img = Image.open('earth.png')
print('Generating lens map')
#lens_map = model.generate_lens_map(lensing.trivial_lens, img.size, (3e+3,3e+7))
#lens_map = model.generate_lens_map(lensing.thin_lens, img.size, (3e+3,3e+7,lensing.snells_law,1.5))
lens_map = model.generate_lens_map(lensing.schwarzschild_thin_lens, img.size, (3e+1,3e+7,S))
#lens_map = model.generate_lens_map(lensing.schwarzschild_lens, img.size, (3e+1,3e+7,S))
print('Applying lens to image')
model.apply_lensing(img, lens_map)
