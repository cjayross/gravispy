import numpy as np
from numpy.linalg import norm
from .geom import Ray, plane_intersect

def snells_law(theta, ref_index):
    return np.arcsin(ref_index*np.sin(theta))

def thin_lens(plane, rays, deflection_function, *args):
    ret = []
    for ray in rays:
        T = plane_intersect(plane, ray)
        if T is np.NaN:
            ret.append(None)
            continue
        RT = ray(T)
        D = ray.origin @ plane.normal
        RP = ray.origin - D*plane.normal
        PT = RT - RP
        #theta = np.arccos(D/norm(RT))
        phi = deflection_function(np.arccos(D/norm(RT)), *args)
        #phi = np.arcsin(ref_index*np.sin(theta))
        #phi = np.arcsin(ref_index*np.sqrt(1-(D/norm(RT)**2)))
        if np.isclose(phi, 0., 1e-10): phi = 0.

    """
        phis = []
        for theta in thetas:
            if theta is None or np.abs(theta) >= np.pi/2:
                phis.append(np.NaN)
                continue
            phi = np.arcsin(ref_index*np.sin(theta))
            if np.isclose(phi, 0, 1e-10): phi = 0
            phis.append(phi)
        return phis
    """

def radial_thin_lens(Plane, ray_list, deflection_function):
    pass
