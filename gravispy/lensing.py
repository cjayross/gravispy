import numpy as np
from warnings import warn
from sympy import lambdify
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import quad
from riccipy import Metric
from gravispy.util import FLOAT_EPSILON, wrap
from gravispy.metrics import SphericalMetric

__all__ = ["spherical_lens"]


def spherical_lens(
    angles, metric, r_obs, r_src, min_radius=FLOAT_EPSILON, unstable_orbits=None
):
    """
    Calculate the deflections by a static spherically symmetric
    gravitational lens by an exact lensing equation.

    This functions assumes that the metric...
    1. must be static
    2. must be spherically symmetric
    3. is centered at the massive object

    Angles are measured with respect to the singularity where
    theta = 0 corresponds to the observer looking away from the singularity.

    """
    if r_src == np.inf:
        warn(
            "infinite source distances may result in unstable calculations",
            RuntimeWarning,
            stacklevel=2,
        )
    if r_src < r_obs:
        warn(
            "performing lensing for sources closer to singularity"
            "than the observer has yet to be tested.",
            RuntimeWarning,
            stacklevel=2,
        )

    if isinstance(metric, Metric) and not isinstance(metric, SphericalMetric):
        metric = SphericalMetric(metric)

    if hasattr(metric, "radius") and (min_radius == FLOAT_EPSILON):
        min_radius = metric.radius

    if hasattr(metric, "unstable_orbits") and (unstable_orbits is None):
        unstable_orbits = metric.unstable_orbits

    S2 = metric.radial_factor
    R2 = metric.angular_factor

    # The following values are used to identify rays that can be skipped.
    res = minimize_scalar(R2, method="bounded", bounds=(r_obs, r_src))
    R_min1 = res["fun"]
    res = minimize_scalar(R2, method="bounded", bounds=(min_radius, r_obs))
    R_min2 = res["fun"]
    delta1 = R_min1 / R2(r_obs)
    delta2 = R_min2 / R2(r_obs)

    def r_dot_sq(r, theta):
        return (R2(r) - R2(r_obs) * np.sin(theta) ** 2) / (S2(r) * R2(r))

    def phi_integrand(r, theta):
        num = R2(r_obs) * S2(r)
        den = R2(r) * (R2(r) - R2(r_obs) * np.sin(theta) ** 2)
        return np.sin(theta) * np.sqrt(num / den)

    err = np.seterr(invalid="ignore")

    for theta in wrap(angles, complement=True):
        if np.isclose(theta, 0.0, atol=FLOAT_EPSILON):
            # This is due to the spherical symmetry.
            yield 0.0
            continue

        if np.sin(theta) ** 2 > delta1:
            # Light ray fails to reach source radius.
            yield np.NaN
            continue

        break_points = [0.0]
        bounds = [min_radius, r_obs]
        # The first element represents multiplicity.
        boundaries = [(1, r_obs, r_src)]

        if np.abs(theta) > np.pi / 2:
            if (R_min2 < 0) or (np.sin(theta) ** 2 < delta2):
                # Light ray fails to reach source radius.
                yield np.NaN
                continue

            if unstable_orbits is not None:
                break_points += list(unstable_orbits)
                bounds[0] = min(unstable_orbits)

            res = brentq(r_dot_sq, *bounds, args=(theta,), full_output=True)
            if not res[1].converged:
                warn(
                    "unresolvable impact parameter encountered",
                    RuntimeWarning,
                    stacklevel=2,
                )
                yield np.NaN
                continue

            r_impact = res[0]

            try:
                S2R2_impact = S2(r_impact) * R2(r_impact)
            except ZeroDivisionError:
                warn(
                    "singularity encountered at impact parameter",
                    RuntimeWarning,
                    stacklevel=2,
                )
                pass

            if not (
                (S2R2_impact is np.NaN)
                or any(np.isclose(S2R2_impact, [0.0, np.inf], atol=FLOAT_EPSILON))
            ):
                boundaries.append((2, r_impact, r_obs))

        phi = 0
        for path in boundaries:
            points = break_points if r_src is not np.inf else None
            res = quad(
                phi_integrand,
                *path[1:],
                args=(theta,),
                points=points,
                epsabs=FLOAT_EPSILON
            )
            phi += path[0] * res[0]

        if phi in (np.NaN, np.Inf):
            warn("undefined phi angle encountered", RuntimeWarning, stacklevel=2)
            yield np.NaN
            continue

        yield phi

    np.seterr(**err)
