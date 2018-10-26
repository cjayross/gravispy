import numpy as np
from numpy.linalg import norm
from .geom import FLOAT_EPSILON, Ray, plane_intersect

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
        D = plane.normal @ (ray.origin - plane.origin)
        phi = deflection_function(np.arccos(D/norm(RT-ray.origin)), *args)
        if np.isclose(phi, 0., atol=FLOAT_EPSILON):
            phi = 0.
        elif phi is np.NaN:
            ret.append(None)
            continue
        new_ray = Ray(RT, plane.normal)
        if phi is not 0:
            new_ray.rotate(phi, np.cross(ray.dir, plane.normal))
        ret.append(new_ray)
    return ret

def radial_thin_lens(plane, rays, deflection_function, *args):
    ret = []
    for ray in rays:
        T = plane_intersect(plane, ray)
        if T is np.NaN:
            ret.append(None)
            continue
        RT = ray(T)
        D = plane.normal @ (ray.origin - plane.origin)
        RP = ray.origin - D*plane.normal
        phi = deflection_function(norm(RP-RT), *args)
        if np.isclose(phi, 0., atol=FLOAT_EPSILON):
            phi = 0.
        elif phi is np.NaN:
            ret.append(None)
            continue
        new_ray = Ray(RT, ray.dir)
        # Note the minus sign during future testing
        if phi is not 0: new_ray.rotate(-phi, np.cross(ray.dir, plane.normal))
        ret.append(new_ray)
    return ret

def spherical_gravity_lens(metric, rays, deflection_function, *args):
    if not metric.assumptions['spherical'] or not metric.assumptions['static']:
        raise ValueError('Metric must be spherically symmetric and static')
    ret = []
    for ray in rays:
        pass
