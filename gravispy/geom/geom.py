import numpy as np
from numpy.linalg import norm

FLOAT_EPSILON = 1e-8

class Ray (object):
    """
    Basic Ray object containing both an origin and a direction.
    Does not contain information on length, however, allows parameterization.

    When called, returns the location of the ray's endpoint if it had the magnitude specified by
    the input parameter.

    Parameters
    ==========
    origin: 3D coordinate specified by either a list, tuple, or ndarray.

    direction: either described by a pair of angles or a 3D coordinate. In either
               case, it must be a list, tuple, or ndarray.
    """
    def __init__(self, origin, direction):
        # type check origin; direction will be checked in the _build method
        if not isinstance(origin, (list, tuple, np.ndarray)) or len(origin) is not 3:
            raise TypeError('origin must be a 3D cartesian coordinate')
        self.origin = np.array(origin, dtype=np.float)
        self._angles = None
        self._dir = None
        if len(direction) is 2:
            self.angles = direction
        elif len(direction) is 3:
            self.dir = direction
        else:
            raise TypeError('direction must be given as either a pair of angles or a 3D vector')

    @property
    def angles(self):
        return self._angles

    @property
    def dir(self):
        return self._dir

    @angles.setter
    def angles(self, direction):
        if len(direction) is 2:
            self._angles = np.array(direction)
            self._dir = np.array([np.sin(self._angles[0])*np.cos(self._angles[1]),
                                  np.sin(self._angles[0])*np.sin(self._angles[1]),
                                  np.cos(self._angles[0])])
            self._dir = self._dir/np.linalg.norm(self._dir)
            # np.isclose helps remedy floating point precision errors.
            for idx, elem in enumerate(self._dir):
                if np.isclose(elem, 0, atol=FLOAT_EPSILON): self._dir[idx] = 0.
        else:
            raise TypeError('direction must be given as a pair of angles')

    @dir.setter
    def dir(self, direction):
        if len(direction) is 3:
            self._dir = np.array(direction)/np.linalg.norm(direction)
            self._angles = np.array([np.arccos(self._dir[2]),
                                     np.arctan(self._dir[1]/self._dir[0])])
            # np.isclose helps remedy floating point precision errors.
            for idx, elem in enumerate(self._angles):
                if np.isclose(elem, 0, atol=FLOAT_EPSILON): self._angles[idx] = 0.
        else:
            raise TypeError('direction must be given as a 3D vector')

    def rotate(self, angle, axis=[0,0,1]):
        self.dir = rotate3D(angle, axis) @ self.dir

    def __call__(self, param):
        return self.dir*param + self.origin

    def __str__(self):
        return '{}(O:{}, D:{})'.format(self.__class__.__name__, self.origin, self.dir)

    def __repr__(self):
        return str(self.__class__)

class Plane (object):
    """
    Represents a plane in 3D space.
    Specified by a plane origin and a normal.

    A plane can be defined by a single Ray argument, otherwise requires two separate 3D vectors.

    We should consider allowing for planes to be confined to a specified height and width.
    """
    def __init__(self, *args):
        if len(args) is 1:
            if not isinstance(args[0], Ray):
                raise TypeError('a single argument must be a Ray')
            self.origin = args[0].origin
            self.normal = args[0].dir
        elif len(args) is 2:
            if (not all(map(isinstance, args, len(args)*((list,tuple,np.ndarray),)))
                    or not all(map(lambda a: len(a) is 3, args))):
                raise TypeError('multiple arguments must be 3D cartesian coordinates')
            self.origin = np.array(args[0])
            self.normal = np.array(args[1])

    def __str__(self):
        return '{}(O:{}, N:{})'.format(self.__class__.__name__, self.origin, self.normal)

    def __repr__(self):
        return str(self.__class__)

class Sphere (object):
    """
    Very simple sphere object; should consider expanding in the near future.
    """
    def __init__(self, origin, radius):
        if not isinstance(origin, (list, tuple, np.ndarray)) or len(origin) is not 3:
            raise TypeError('origin must be a 3D cartesian coordinate')
        if not isinstance(radius, (int, float)):
            raise TypeError('radius must be a numerical value')
        self.origin = origin
        self.radius = radius

    def __str__(self):
        return '{}(O:{}, R:{})'.format(self.__class__.__name__, self.origin, self.radius)

    def __repr__(self):
        return str(self.__class__)

def plane_intersect(plane, ray):
    """
    Returns the ray parameter associated with the ray's intersection with a plane.
    """
    numer = (plane.origin - ray.origin) @ plane.normal
    denom = ray.dir @ plane.normal
    if numer == 0 or denom == 0: return np.NaN
    ret = numer/denom
    return ret if ret > 0 else np.NaN

def sphere_intersect(ray, sphere):
    """
    Returns the ray parameter associated with the ray's intersection with a sphere.
    """
    OS = ray.origin - sphere.origin
    B = 2*ray.dir @ OS
    C = OS @ OS - sphere.radius**2
    disc = B**2 - 4*C
    if disc > 0:
        dist = np.sqrt(disc)
        Q = (-B - dist)/2 if B < 0 else (-B + dist)/2
        ret0, ret1 = sorted([Q, C/Q])
        if not ret1 < 0: return ret1 if ret0 < 0 else ret0
    return np.NaN

def rotate3D(angle, axis=[0,0,1]):
    axis = np.array(axis)/norm(axis)
    return (np.cos(angle)*np.eye(3)
            + np.sin(angle)*np.array([[0, -axis[2], axis[1]],[axis[2], 0, -axis[0]],[-axis[1], axis[0], 0]])
            + (1-np.cos(angle))*np.outer(axis, axis))
