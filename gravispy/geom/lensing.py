import numpy as np
from warnings import warn
from numpy.linalg import norm
from scipy.optimize import fsolve, minimize_scalar
from scipy.integrate import quad
from .geom import FLOAT_EPSILON, Plane, Ray, NullRay, plane_intersect
from .metric import SphericalSpacetime, BarriolaVilenkin, EllisWormhole,\
                    Schwarzschild

__all__ = [
        'snells_law',
        'schwarzschild_deflection',
        'thin_lens',
        'radial_thin_lens',
        'schwarzschild_thin_lens',
        'static_spherical_lens',
        'barriola_vilenkin_lens',
        'ellis_wormhole_lens',
        ]

### Deflection functions ###

def snells_law(theta, ref_index):
    return np.arcsin(ref_index*np.sin(theta))

def schwarzschild_deflection(r, metric):
    if not isinstance(metric, Schwarzschild):
        raise TypeError('metric must be Schwarzschild')
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

def static_spherical_lens(rays, rS, metric):
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

    S2 = metric.radial_factor(generator=True)
    R2 = metric.angular_factor(generator=True)

    # Stores infimum calculations for rays of common origins
    R_infs = dict()

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
            warn('unable to resolve sources closer to'
                 'singularity than observer',
                 RuntimeWarning, stacklevel=2)
            yield NullRay([0,0,0])
            continue
        # angle with the optical axis (0 => phi = 0)
        # optical axis = ray.origin
        # the domain of theta is [0, pi] with it's sign determined
        # by the cross product of the final rotation (hopefully).
        theta = np.arccos(ray.dir @ (ray.origin/rO))
        if np.isclose(theta, 0., atol=FLOAT_EPSILON):
            # by symmetry
            yield Ray([0,0,0], ray.origin)
            continue

        # minimize_scalar is an expensive function;
        # this reduces its usage.
        # when len(rays) == 1000 (of the same origin), this reduces
        # the execution time from 3 seconds to about 0.001 seconds.
        # dictionaries are definitely OP. Please nerf.
        if rO not in R_infs.keys():
            R_inf1 = minimize_scalar(
                    R2,
                    method='bounded',
                    bounds=(rO, rS))['fun']
            R_inf2 = minimize_scalar(
                    R2,
                    method='bounded',
                    bounds=(0, rO))['fun']
            R_infs.update({rO:(R_inf1, R_inf2)})

        if np.sin(theta)**2 > (R_infs[rO][0]/R2(rO)):
            # the light ray fails to reach rS
            yield NullRay([0,0,0])
            continue

        break_points = [0.]
        # note: the first element represents multiplicity
        boundaries = [(1, rO, rS)]

        if ray.dir @ (ray.origin/rO) < 0:
            if hasattr(metric, 'unstable_orbits'):
                break_points += list(metric.unstable_orbits)

            if R_infs[rO][1] < 0 or np.sin(theta)**2 < (R_infs[rO][1]/R2(rO)):
                # the light ray fails to reach rS
                yield NullRay([0,0,0])
                continue

            # attempt to find an impact parameter with 5 random step factors
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

            # the impact parameter must be positive
            if max(fsolve_res[0]) > 0.:
                rP = max(fsolve_res[0])
                break_points.append(rP)
                if (S2(rP)*R2(rP) is np.NaN
                        or any(np.isclose(S2(rP)*R2(rP), [0., np.inf],
                                          atol=FLOAT_EPSILON))):
                    # rP is a singularity
                    pass
                elif rP <= rO:
                    boundaries.append((2, rP, rO))
                else:
                    # TODO, investigate the possibility of this result
                    warn('unresolved fsolve result encountered',
                         RuntimeWarning, stacklevel=2)
                    yield NullRay([0,0,0])
                    continue

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

        if phi is np.NaN or phi is np.Inf:
            warn('unresolvable integration result',
                 RuntimeWarning, stacklevel=2)
            yield NullRay([0,0,0])
            continue

        new_ray = Ray([0,0,0], ray.origin)
        if not np.isclose(phi, 0., atol=FLOAT_EPSILON):
            new_ray.rotate(phi, np.cross(new_ray.dir, ray.dir))
        yield new_ray

def barriola_vilenkin_lens(rays, rS, metric):
    if not isinstance(metric, BarriolaVilenkin):
        raise TypeError('metric must describe a Barriola-Vilenkin spacetime')
    if any(map(lambda a: a not in metric.basis, metric.args)):
        raise ValueError('metric has unset variables')

    for ray in rays:
        rO = norm(ray.origin)
        theta = np.arccos(ray.dir @ (ray.origin/rO))
        if rS < rO:
            warn('unable to resolve sources closer to'
                 'singularity than observer',
                 RuntimeWarning, stacklevel=2)
            yield NullRay([0,0,0])
            continue

        phi = (theta - np.arcsin(rO*np.sin(theta)/rS))/metric.k

        new_ray = Ray([0,0,0], ray.origin)
        if not np.isclose(phi, 0., atol=FLOAT_EPSILON):
            new_ray.rotate(phi, np.cross(new_ray.dir, ray.dir))
        yield new_ray

def ellis_wormhole_lens(rays, rS, metric, orient=1.):
    """
    Work in progres...
    """
    if not isinstance(metric, EllisWormhole):
        raise TypeError('metric must describe an Ellis wormhole')
    if any(map(lambda a: a not in metric.basis, metric.args)):
        raise ValueError('metric has unset variables')

    def phi_func(r, rO, theta):
        return np.sqrt((rO**2+metric.a**2)
                       / ((r**2+metric.a**2)
                         * (r**2
                            + metric.a**2*np.cos(theta)**2
                            - rO**2*np.sin(theta)**2)))\
               * np.sin(theta)

    for ray in rays:
        rO = orient*norm(ray.origin)
        theta = np.arccos(ray.dir @ (ray.origin/rO))

        if (np.sin(theta)**2 >= metric.a**2 / (rO**2+metric.a**2)
                or np.abs(theta) >= np.pi/2):
            yield NullRay([0,0,0])
            continue

        phi = quad(
                phi_func,
                rO, rS,
                (rO, theta),
                epsabs=FLOAT_EPSILON,
                )[0]

        new_ray = Ray([0,0,0], orient*ray.origin)
        if not np.isclose(phi, 0., atol=FLOAT_EPSILON):
            new_ray.rotate(phi, np.cross(new_ray.dir, ray.dir))
        yield new_ray
