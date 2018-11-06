import numpy as np
from numpy.linalg import norm
from scipy.optimize import fsolve
from scipy.integrate import quad
from .geom import FLOAT_EPSILON, Plane, Ray, NullRay, plane_intersect
from .metric import SphericalSpacetime

__all__ = [
        'snells_law',
        'schwarzschild_deflection',
        'thin_lens',
        'radial_thin_lens',
        'schwarzschild_thin_lens',
        ]

### Deflection functions ###

def snells_law(theta, ref_index):
    return np.arcsin(ref_index*np.sin(theta))

def schwarzschild_deflection(r, metric):
    if hasattr(metric.mass, 'is_number') and not metric.mass.is_number:
        raise ValueError('Schwarzschild mass not set.')
    if r <= metric.radius: return np.NaN
    return 2*float(metric.radius)/r

### Lensing functions ###

def thin_lens(plane, rays, deflection_function, *args):
    for ray in rays:
        T = plane_intersect(plane, ray)
        if T is np.NaN:
            yield NullRay()
            continue
        RT = ray(T)
        D = plane.normal @ (ray.origin - plane.origin)
        phi = deflection_function(np.arccos(D/norm(RT-ray.origin)), *args)
        if phi is np.NaN:
            yield NullRay(RT)
            continue
        new_ray = Ray(RT, -np.sign(D)*plane.normal)
        if not np.isclose(phi, 0., atol=FLOAT_EPSILON):
            new_ray.rotate(phi, np.cross(new_ray.dir, ray.dir))
        yield new_ray

def radial_thin_lens(plane, rays, deflection_function, *args):
    for ray in rays:
        T = plane_intersect(plane, ray)
        if T is np.NaN:
            yield NullRay()
            continue
        RT = ray(T)
        D = plane.normal @ (ray.origin - plane.origin)
        RP = ray.origin - D*plane.normal
        phi = deflection_function(norm(RP-RT), *args)
        if phi is np.NaN:
            yield NullRay(RT)
            continue
        new_ray = Ray(RT, ray.dir)
        if not np.isclose(phi, 0., atol=FLOAT_EPSILON):
            new_ray.rotate(
                    phi,
                    np.cross(new_ray.dir, -np.sign(D)*plane.normal))
        yield new_ray

def schwarzschild_thin_lens(rays, metric):
    return radial_thin_lens(Plane([0,0,0], rays[0].origin), rays,
                            schwarzschild_deflection, metric)

def static_spherical_grav_lens(rays, rS, metric):
    """
    Work in progress...

    This functions assumes that the metric...
    1. must be static
    2. must be spherically symmetric
    3. is centered at the massive object
    4. is aligned such that the optical axis coincides with the x-axis and that
       the observer is situated at phi = 0.

    And assumes that the rays each have origins with respect to
    the massive object.
    """
    if not isinstance(metric, SphericalSpacetime):
        raise TypeError('metric must be a spherically symmetric spacetime')
    if not metric.assumptions['static']:
        raise ValueError('metric must be static')
    if any(map(lambda a: a not in metric.basis, metric.args)):
        raise ValueError('metric has unset variables')

    # A(r)**2 - metric conformal factor
    # S(r)**2 - metric radial factor
    # R(r)**2 - metric angular factor
    A2 = metric.conformal_factor(generator=True)
    S2 = metric.radial_factor(generator=True)
    R2 = metric.angular_factor(generator=True)

    def impact_func(r, rO, theta):
        return (R2(r)-R2(rO)*np.sin(theta)**2) / (S2(r)*R2(r))

    def phi_func(r, rO, theta):
        return np.sqrt(R2(rO)*S2(r)
                       / (R2(r)*(R2(r)-R2(rO)*np.sin(theta)**2)))\
               * np.sin(theta)

    for ray in rays:
        # radial position of observer
        rO = norm(ray.origin)
        if rS < rO:
            yield NullRay([0,0,0])
            continue
        # impact parameter
        rP = None
        # angle with the optical axis (0 => phi = 0)
        # optical axis = [1,0,0]
        # ray.dir @ [1,0,0] = ray.dir[0] (x-value of ray.dir)
        # ray.dir is normalized so abs(ray.dir[0]) <= 1
        # the domain of theta is [-pi, pi]
        #sign = np.sign(ray.dir[0]) if ray.dir[0] != 0. else 1.
        theta = np.arccos(ray.dir[0])
        if np.isclose(theta, 0., atol=FLOAT_EPSILON):
            # by symmetry
            yield Ray([0,0,0],[1,0,0])
            continue
        # sign of the output source angle, phi
        break_points = [0.]
        # note: the first element represents multiplicity
        boundaries = [(1, rO, rS)]

        if hasattr(metric, 'unstable_orbits') and ray.dir[0] < 0:
            break_points += list(metric.unstable_orbits)
            fsolve_res = fsolve(
                    impact_func,
                    metric.unstable_orbits,
                    (rO, theta),
                    full_output=True,
                    xtol=FLOAT_EPSILON,
                    factor=0.1,
                    )
            if fsolve_res[2] is not 1 or max(fsolve_res[0]) < 0.:
                # solution did not converge on any positive roots
                pass
            else:
                try:
                    rP = max(fsolve_res[0])
                    1/(S2(rP)*R2(rP))
                    break_points.append(rP)
                    if rP < rO:
                        boundaries.append((2, rP, rO))
                    elif rP >= rS:
                        boundaries.append((2, rS, rP))
                    else:
                        # the lightray never reaches rS
                        yield NullRay([0,0,0])
                        continue

                except ZeroDivisionError:
                    # rP is a singularity
                    pass
        phi = 0
        for path in boundaries:
            integral = quad(
                    phi_func,
                    *path[1:],
                    (rO, theta),
                    points=break_points,
                    epsabs=FLOAT_EPSILON,
                    )
            phi += path[0] * integral[0]

        if phi is np.NaN or phi is np.Inf:
            yield NullRay([0,0,0])
            continue
        new_ray = Ray([0,0,0],[1,0,0])
        if not np.isclose(phi, 0., atol=FLOAT_EPSILON):
            new_ray.rotate(phi, np.cross(new_ray.dir, ray.dir))
        yield new_ray
