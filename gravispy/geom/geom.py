import numpy as np
from numpy.linalg import norm

__all__ = [
        'FLOAT_EPSILON',
        'constants',
        'geom_units',
        'Ray',
        'NullRay',
        'Plane',
        'Sphere',
        'plane_intersect',
        'sphere_intersect',
        'rotate3D',
        ]

# This error tolerance will maintain accuracy up to 10 meters in units of c
# or about 5 kilometers in astronomical units
FLOAT_EPSILON = 3.3356409519815204e-08
constants = {
        'c': 299792458.0,
        'G': 6.67408e-11,
        'gravitational constant': 6.67408e-11,
        'solar_mass': 1.98847e+30,
        'au': 149597870691.0,
        'astronomical unit': 149597870691.0,
        'parsec': 3.0856775813057292e+16,
        'ly': 9460730472580800.0,
        'light year': 9460730472580800.0,
        }
# geometrized units
geom_units = {
        'm': 1.0,
        'length': 1.0,
        's': 299792458.0,
        'time': 299792458.0,
        'kg': 7.425915486106335e-28, # G/c**2
        'mass': 7.425915486106335e-28,
        'velocity': 3.3356409519815204e-09, # 1/c
        'acceleration': 1.1126500560536185e-17, # 1/c**2
        'energy': 8.262445281865645e-45, # G/c**4
        }

class Ray (object):
    """
    Basic Ray object containing both an origin and a direction.
    Does not contain information on length, however, allows parameterization.

    When called, returns the location of the ray's endpoint if it had the
    magnitude specified by the input parameter.

    Parameters
    ==========
    origin: iterable
        3D coordinate specified by either a list, tuple, or ndarray.
    direction: iterable
        Either described by a pair of angles or a 3D coordinate.
        In either case, it must be a list, tuple, or ndarray.
    """
    def __init__(self, origin, direction):
        # type check origin; direction will be checked in the _build method
        if (not isinstance(origin, (list, tuple, np.ndarray))
                or len(origin) is not 3):
            raise TypeError('origin must be a 3D cartesian coordinate')
        self.origin = np.array(origin, dtype=np.float)
        self._angles = None
        self._dir = None
        if len(direction) is 2:
            self.angles = direction
        elif len(direction) is 3:
            self.dir = direction
        else:
            raise TypeError('direction must be given as either a pair of '\
                            'angles or a 3D vector')

    @property
    def angles(self):
        return self._angles

    @property
    def dir(self):
        return self._dir

    @angles.setter
    def angles(self, direction):
        if len(direction) is 2:
            mod_theta = np.clip(direction[0],0,np.pi)
            mod_phi = (direction[1]+np.pi) % (2*np.pi) - np.pi
            self._angles = np.array([mod_theta, mod_phi])
            self._dir = np.array([
                np.sin(self._angles[0])*np.cos(self._angles[1]),
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
                                     np.arctan2(self._dir[1], self._dir[0])])
            # np.isclose helps remedy floating point precision errors.
            for idx, elem in enumerate(self._angles):
                if np.isclose(elem, 0, atol=FLOAT_EPSILON):
                    self._angles[idx] = 0.
        else:
            raise TypeError('direction must be given as a 3D vector')

    def rotate(self, angle, axis=[0,0,1]):
        self.dir = rotate3D(angle, axis) @ self.dir

    def __call__(self, param):
        return self.dir*param + self.origin

    def __str__(self):
        return '{}(O:{}, D:{})'.format(
                self.__class__.__name__, self.origin, self.angles)

    def __repr__(self):
        return str(self.__class__)

class NullRay (Ray):
    def __init__(self, origin=3*[np.NaN]):
        super(NullRay, self).__init__(origin, 2*[np.NaN])

class Plane (object):
    """
    Represents a plane in 3D space.
    Specified by a plane origin and a normal.

    A plane can be defined by a single Ray argument, otherwise requires
    two separate 3D vectors.

    We should consider allowing for planes to be confined to a specified
    height and width.
    """
    def __init__(self, *args):
        if len(args) is 1:
            if not isinstance(args[0], Ray):
                raise TypeError('a single argument must be a Ray')
            self.origin = args[0].origin
            self.normal = args[0].dir
        elif len(args) is 2:
            # I really abuse the hell out of all(map(...))
            if (not all(map(isinstance, args,
                            len(args)*((list,tuple,np.ndarray),)))
                    or not all(map(lambda a: len(a) is 3, args))):
                raise TypeError('multiple arguments must be '\
                                '3D cartesian coordinates')
            self.origin = np.array(args[0])
            self.normal = np.array(args[1])/norm(args[1])

    def __str__(self):
        return '{}(O:{}, N:{})'.format(
                self.__class__.__name__, self.origin, self.normal)

    def __repr__(self):
        return str(self.__class__)

class Sphere (object):
    """
    Very simple sphere object; should consider expanding in the near future.
    """
    def __init__(self, origin, radius):
        if (not isinstance(origin, (list, tuple, np.ndarray))
                or len(origin) is not 3):
            raise TypeError('origin must be a 3D cartesian coordinate')
        if not isinstance(radius, (int, float)):
            raise TypeError('radius must be a numerical value')
        self.origin = origin
        self.radius = radius

    def __str__(self):
        return '{}(O:{}, R:{})'.format(
                self.__class__.__name__, self.origin, self.radius)

    def __repr__(self):
        return str(self.__class__)

def plane_intersect(plane, ray):
    """
    Returns the ray parameter associated with the ray's intersection
    with a plane.
    """
    numer = (plane.origin - ray.origin) @ plane.normal
    denom = ray.dir @ plane.normal
    if numer == 0 or denom == 0: return np.NaN
    ret = numer/denom
    return ret if ret > 0 else np.NaN

def sphere_intersect(ray, sphere):
    """
    Returns the ray parameter associated with the ray's intersection
    with a sphere.
    """
    OS = ray.origin - sphere.origin
    B = 2*ray.dir @ OS
    C = OS @ OS - sphere.radius**2
    disc = B**2 - 4*C
    if disc > 0:
        dist = np.sqrt(disc)
        Q = (-B - dist)/2 if B < 0 else (-B + dist)/2
        ret0, ret1 = sorted([Q, C/Q])
        if ret1 >= 0: return ret1 if ret0 < 0 else ret0
    return np.NaN

def rotate3D(angle, axis=[0,0,1]):
    axis = np.array(axis)/norm(axis)
    return (np.cos(angle)*np.eye(3)
            + np.sin(angle)*np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]])
            + (1-np.cos(angle))*np.outer(axis, axis))
