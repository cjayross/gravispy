import numpy as np
from numpy.linalg import norm
from .geom import FLOAT_EPSILON, Plane, Ray, NullRay, plane_intersect

### Deflection functions ###

def snells_law(theta, ref_index):
    return np.arcsin(ref_index*np.sin(theta))

def schwarzschild_deflection(r, metric):
    if hasattr(metric.mass, 'is_number') and not metric.mass.is_number:
        raise ValueError('Schwarzschild mass not set.')
    if r <= metric.radius(): return np.NaN
    return 2*metric.radius()/r

### Lensing functions ###

def thin_lens(plane, rays, deflection_function, *args):
    ret = []
    for ray in rays:
        T = plane_intersect(plane, ray)
        if T is np.NaN:
            ret.append(NullRay())
            continue
        RT = ray(T)
        D = plane.normal @ (ray.origin - plane.origin)
        phi = deflection_function(np.arccos(D/norm(RT-ray.origin)), *args)
        if np.isclose(phi, 0., atol=FLOAT_EPSILON):
            phi = 0.
        elif phi is np.NaN:
            ret.append(NullRay(RT))
            continue
        new_ray = Ray(RT, -np.sign(D)*plane.normal)
        if phi is not 0:
            new_ray.rotate(phi, np.cross(new_ray.dir, ray.dir))
        ret.append(new_ray)
    return ret

def radial_thin_lens(plane, rays, deflection_function, *args):
    ret = []
    for ray in rays:
        T = plane_intersect(plane, ray)
        if T is np.NaN:
            ret.append(NullRay())
            continue
        RT = ray(T)
        D = plane.normal @ (ray.origin - plane.origin)
        RP = ray.origin - D*plane.normal
        phi = deflection_function(norm(RP-RT), *args)
        if np.isclose(phi, 0., atol=FLOAT_EPSILON):
            phi = 0.
        elif phi is np.NaN:
            ret.append(NullRay(RT))
            continue
        new_ray = Ray(RT, ray.dir)
        # Note the minus sign during future testing
        if phi is not 0:
            new_ray.rotate(phi, np.cross(new_ray.dir, -np.sign(D)*plane.normal))
        ret.append(new_ray)
    return ret

def schwarzschild_thin_lens(rays, metric):
    lens = Plane([0,0,0], rays[0].origin)
    return radial_thin_lens(lens, rays, schwarzschild_deflection, metric)
