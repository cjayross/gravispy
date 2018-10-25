import numpy as np
from numpy.linalg import norm
from .geom import Ray, plane_intersect

def thin_lens(plane, ray_list, deflection_function):
    ret = []
    for ray in ray_list:
        T = plane_intersect(plane, ray)
        if T is np.NaN:
            ret.append(None)
            continue

def radial_thin_lens(plane, ray_list, deflection_function):
    pass
