import numpy as np
from itertools import repeat
from sympy import ones, lambdify, sin
from riccipy.metric import load_metric, SpacetimeMetric
from gravispy.util import FLOAT_EPSILON

__all__ = [
    "is_static_spherical",
    "SphericalMetric",
    "Schwarzschild",
]


def is_static_spherical(metric):
    con = metric[0, 0]
    rad = metric[1, 1]
    ang = metric[2, 2]
    azi = metric[3, 3] / sin(metric.coords[2]) ** 2
    symbol_sets = map(getattr, [con, rad, ang, azi], repeat("free_symbols"))
    res = set.intersection(set.union(*symbol_sets), metric.coords)
    return (len(res) == 1) and (metric.coords[1] in res)


class SphericalMetric(object):
    def __new__(cls, obj):
        if not is_static_spherical(obj):
            raise ValueError(
                "metric object is either not spherically symmetric,"
                "not static, or is not in spherical coordinates"
            )
        extra_variables = set.difference(obj.as_array().free_symbols, obj.coords)
        if len(extra_variables) != 0:
            raise ValueError("metric has unset variables: {}".format(extra_variables))
        sign = 1 if obj.is_timelike else -1
        conformal = sign * obj[0, 0]
        radial = -sign * obj[1, 1] / conformal
        angular = -sign * obj[2, 2] / conformal
        obj.conformal_factor = lambdify(obj.coords[1], conformal)
        obj.radial_factor = lambdify(obj.coords[1], radial)
        obj.angular_factor = lambdify(obj.coords[1], angular)
        return obj


class Schwarzschild(SpacetimeMetric, SphericalMetric):
    def __new__(cls, mass=None):
        metric, var, _ = load_metric(
            "g", "schwarzschild", coords="spherical", timelike=True
        )
        obj = SpacetimeMetric.__new__(cls, "g", metric.coords, metric.as_array())
        obj.mass = mass
        obj.set_variables({var: mass})
        obj = SphericalMetric.__new__(cls, obj)
        return obj

    @property
    def radius(self):
        return 2 * self.mass

    @property
    def unstable_orbits(self):
        return np.array([3 * self.mass, 6 * self.mass])
