import numpy as np
from sympy import ones, lambdify
from riccipy.metric import load_metric, SpacetimeMetric
from gravispy.util import FLOAT_EPSILON

__all__ = [
    'SphericalMetric',
    'Schwarzschild',
]

class SphericalMetric (object):
    def __new__(cls, obj):
        sign = 1 if obj.is_timelike else -1
        conformal = sign * obj[0,0]
        radial = -sign * obj[1,1] / conformal
        angular = -sign * obj[2,2] / conformal
        obj.conformal_factor = lambdify(obj.coords[1], conformal)
        obj.radial_factor = lambdify(obj.coords[1], radial)
        obj.angular_factor = lambdify(obj.coords[1], angular)
        return obj

    @property
    def min_radius(self):
        return FLOAT_EPSILON

class Schwarzschild (SpacetimeMetric, SphericalMetric):
    def __new__(cls, mass=None):
        metric, var, _ = load_metric('g', 'schwarzschild', coords='spherical', timelike=True)
        obj = SpacetimeMetric.__new__(cls, 'g', metric.coords, metric.as_array())
        if mass is not None:
            obj.mass = mass
            obj.set_variables({var: mass})
        else:
            obj.mass = var
        obj = SphericalMetric.__new__(cls, obj)
        return obj

    @property
    def radius(self):
        return 2*self.mass

    @property
    def min_radius(self):
        return self.radius

    @property
    def unstable_orbits(self):
        return np.array([3*self.mass, 6*self.mass])
