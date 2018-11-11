import numpy as np
from warnings import warn
from numpy.linalg import norm
from scipy.optimize import fsolve, minimize_scalar
from scipy.integrate import quad
from .geom import FLOAT_EPSILON, Plane, Sphere, Ray, NullRay,\
                  unwrap, plane_intersect, sphere_intersect
from .metric import SphericalSpacetime, BarriolaVilenkin, EllisWormhole,\
                    Schwarzschild

__all__ = [
        'trivial_deflection',
        'snells_law_deflection',
        'schwarzschild_deflection',
        'thin_lens',
        'trivial_lens',
        'radial_thin_lens',
        'schwarzschild_thin_lens',
        'static_spherical_lens',
        'barriola_vilenkin_lens',
        'ellis_wormhole_lens',
        ]

### Deflection functions ###

def trivial_deflection(theta):
    return theta

def snells_law_deflection(theta, ref_index):
    return np.arcsin(ref_index*np.sin(theta))

def schwarzschild_deflection(r, metric):
    if not isinstance(metric, Schwarzschild):
        raise TypeError('metric must be Schwarzschild')
    if hasattr(metric.mass, 'is_number') and not metric.mass.is_number:
        raise ValueError('Schwarzschild mass not set.')
    if r <= metric.radius: return np.NaN
    return 2*float(metric.radius)/r

### Lensing functions ###

def thin_lens(angles, rO, rS, deflection_function, *args):
    if isinstance(angles, (float, int)):
        angles = np.array([angles])
    plane = Plane([0,0,0], [rO,0,0])
    sphere = Sphere([0,0,0], rS)
    for theta in angles:
        ray = Ray([rO,0,0], [np.pi/2,theta])
        T = plane_intersect(plane, ray)
        if T is np.NaN:
            yield unwrap(theta)
            continue
        RT = ray(T)
        D = plane.normal @ (ray.origin - plane.origin)
        phi = deflection_function(np.arccos(D/norm(RT-ray.origin)), *args)
        if phi is np.NaN:
            yield np.NaN
            continue
        new_ray = Ray(RT, -np.sign(D)*plane.normal)
        if not np.isclose(phi, 0., atol=FLOAT_EPSILON):
            new_ray.rotate(phi, np.cross(new_ray.dir, ray.dir))
        RT = new_ray(sphere_intersect(new_ray, sphere))
        yield np.arctan2(RT[1], RT[0])

def trivial_lens(angles, rO, rS):
    return thin_lens(angles, rO, rS, trivial_deflection)

def radial_thin_lens(angles, rO, rS, deflection_function, *args):
    if isinstance(angles, (float, int)):
        angles = np.array([angles])
    plane = Plane([0,0,0], [rO,0,0])
    sphere = Sphere([0,0,0], rS)
    for theta in angles:
        ray = Ray([rO,0,0], [np.pi/2,theta])
        T = plane_intersect(plane, ray)
        if T is np.NaN:
            yield unwrap(theta)
            continue
        RT = ray(T)
        D = plane.normal @ (ray.origin - plane.origin)
        RP = ray.origin - D*plane.normal
        phi = deflection_function(norm(RP-RT), *args)
        if phi is np.NaN:
            yield np.NaN
            continue
        new_ray = Ray(RT, ray.dir)
        if not np.isclose(phi, 0., atol=FLOAT_EPSILON):
            new_ray.rotate(
                    phi, np.cross(new_ray.dir, -np.sign(D)*plane.normal))
        RT = new_ray(sphere_intersect(new_ray, sphere))
        yield np.arctan2(RT[1], RT[0])

def schwarzschild_thin_lens(angles, rO, rS, metric):
    return radial_thin_lens(angles, rO, rS, schwarzschild_deflection, metric)

def static_spherical_redshift(rO, rS, metric):
    """
    Returns the ratio of the frequencies between a photon that is emitted
    and that is recieved through the effects of a static, spherical spacetime.
    This is given by the conformal factor of the metric as,

    \frac{\omega_O}{\omega_S} = \frac{A(r_S)}{A(r_O)}.
    """
    if not isinstance(metric, SphericalSpacetime):
        raise TypeError('metric must be a spherically symmetric spacetime')
    if not metric.assumptions['static']:
        raise ValueError('metric must be static')
    if any(map(lambda a: a not in metric.basis, metric.args)):
        raise ValueError('metric has unset variables')

    A2 = metric.conformal_factor(generator=True)
    return np.sqrt(A2(rO)/A2(rS))

def static_spherical_lens(angles, rO, rS, metric):
    """
    Calculate the deflections by a static spherically symmetric
    gravitational lens by an exact lensing equation.

    This functions assumes that the metric...
    1. must be static
    2. must be spherically symmetric
    3. is centered at the massive object

    And assumes that the rays each have origins with respect to
    the massive object.
    """
    if not isinstance(metric, SphericalSpacetime):
        raise TypeError('metric must be a spherically symmetric spacetime')
    if not metric.assumptions['static']:
        raise ValueError('metric must be static')
    if any(map(lambda a: a not in metric.basis, metric.args)):
        raise ValueError('metric has unset variables')
    if rS is np.inf:
        warn('infinite source distances may result in unstable calculations',
             RuntimeWarning, stacklevel=2)
    # consider removing
    if rS < rO:
        warn('unable to resolve sources closer to'
             'singularity than observer',
             RuntimeWarning, stacklevel=2)
        return len(angles)*[np.NaN]
    if isinstance(angles, (float, int)):
        angles = np.array([angles])

    angles = unwrap(angles)
    S2 = metric.radial_factor(generator=True)
    R2 = metric.angular_factor(generator=True)
    # used to identify possible angles
    R_inf1 = minimize_scalar(
            R2,
            method='bounded',
            bounds=(rO, rS))['fun']
    R_inf2 = minimize_scalar(
            R2,
            method='bounded',
            bounds=(0, rO))['fun']

    def impact_func(r, rO, theta):
        return (R2(r)-R2(rO)*np.sin(theta)**2) / (S2(r)*R2(r))

    def phi_func(r, rO, theta):
        return np.sqrt(R2(rO)*S2(r)
                       / (R2(r)*(R2(r)-R2(rO)*np.sin(theta)**2)))\
               * np.sin(theta)

    for theta in angles:
        if np.isclose(theta, 0., atol=FLOAT_EPSILON):
            # by symmetry
            yield 0.
            continue

        if np.sin(theta)**2 > (R_inf1/R2(rO)):
            # the light ray fails to reach rS
            yield np.NaN
            continue

        break_points = [0.]
        # note: the first element represents multiplicity
        boundaries = [(1, rO, rS)]

        if np.abs(theta) > np.pi/2:
            if R_inf2 < 0 or np.sin(theta)**2 < (R_inf2/R2(rO)):
                # the light ray fails to reach rS
                yield np.NaN
                continue

            if hasattr(metric, 'unstable_orbits'):
                break_points += list(metric.unstable_orbits)

            # attempt to find an impact parameter with 10 random step factors
            # should fsolve fail for all generated factors,
            # artifacts will occur in the output over a continuous domain
            for _ in range(10):
                fsolve_res = fsolve(
                        impact_func,
                        break_points[1:] + [rO],
                        (rO, theta),
                        full_output=True,
                        xtol=FLOAT_EPSILON,
                        factor=(np.random.rand()+0.1) % 1,
                        )
                if fsolve_res[2] is 1:
                    # fsolved managed to converge
                    break
            else: # for-else condition (only executes if loop completes)
                warn('unresolved fsolve result encountered',
                     RuntimeWarning, stacklevel=2)
                yield np.NaN
                continue

            # the impact parameter must be positive
            rP = max(fsolve_res[0])
            if rP > 0.:
                break_points.append(rP)
                if not (S2(rP)*R2(rP) is np.NaN
                        or any(np.isclose(S2(rP)*R2(rP), [0., np.inf],
                                          atol=FLOAT_EPSILON))):
                    # TODO: consider whether cases where
                    # rP > rO should be considered
                    boundaries.append((2, rP, rO))

        phi = 0
        for path in boundaries:
            integral = quad(
                    phi_func,
                    *path[1:],
                    (rO, theta),
                    points=break_points if rS is not np.inf else None,
                    epsabs=FLOAT_EPSILON,
                    )
            phi += path[0] * integral[0]

        if phi in (np.NaN, np.Inf):
            warn('unresolvable integration result',
                 RuntimeWarning, stacklevel=2)
            yield np.NaN
            continue

        yield unwrap(phi)

def barriola_vilenkin_lens(angles, rO, rS, metric):
    if not isinstance(metric, BarriolaVilenkin):
        raise TypeError('metric must describe a Barriola-Vilenkin spacetime')
    if any(map(lambda a: a not in metric.basis, metric.args)):
        raise ValueError('metric has unset variables')
    if rS < rO:
        warn('unable to resolve sources closer to'
             'singularity than observer',
             RuntimeWarning, stacklevel=2)
        return len(angles)*[np.NaN]
    if isinstance(angles, (float, int)):
        angles = np.array([angles])

    angles = unwrap(angles)

    for theta in angles:
        yield unwrap((theta - np.arcsin(rO*np.sin(theta)/rS))/metric.k)

def ellis_wormhole_lens(angles, rO, rS, metric):
    """
    Work in progres...
    """
    if not isinstance(metric, EllisWormhole):
        raise TypeError('metric must describe an Ellis wormhole')
    if any(map(lambda a: a not in metric.basis, metric.args)):
        raise ValueError('metric has unset variables')
    if isinstance(angles, (float, int)):
        angles = np.array([angles])

    angles = unwrap(angles)

    def phi_func(r, rO, theta):
        return np.sqrt((rO**2+metric.a**2)
                       / ((r**2+metric.a**2)
                          * (r**2
                             + metric.a**2*np.cos(theta)**2
                             - rO**2*np.sin(theta)**2)))\
               * np.sin(theta)

    for theta in angles:
        if (np.sin(theta)**2 >= metric.a**2 / (rO**2+metric.a**2)
                or np.abs(theta) >= np.pi/2):
            yield np.NaN
            continue
        yield unwrap(quad(phi_func,rO,rS,(rO,theta),epsabs=FLOAT_EPSILON)[0])
