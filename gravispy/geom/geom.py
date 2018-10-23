import numpy as np

class Ray (object):
    def __init__(self, origin, direction):
        if not isinstance(origin, (list, tuple, np.ndarray)) or len(origin) is not 3:
            raise TypeError('origin must be a 3D cartesian coordinate')
        if not isinstance(direction, (list, tuple, np.ndarray)) or len(direction) not in (2,3):
            raise TypeError('direction must be given as either a pair of angles or a 3D vector')

        self.origin = np.array(origin)
        if len(direction) is 2:
            self.angles = np.array(direction)
            self.dir = np.array([np.sin(self.angles[0])*np.cos(self.angles[1]),
                                 np.sin(self.angles[0])*np.sin(self.angles[1]),
                                 np.cos(self.angles[0])])
            self.dir = self.dir/np.linalg.norm(self.dir)
        else:
            self.dir = np.array(direction)/np.linalg.norm(direction)
            self.angles = np.array([np.arccos(self.dir[2]),
                                    np.arctan(self.dir[1]/self.dir[0])])

    def __call__(self, param):
        return self.dir*param + self.origin

    def __str__(self):
        return '{}(O:{}, D:{})'.format(self.__class__.__name__, self.origin, self.dir)

    def __repr__(self):
        return str(self.__class__)

class Plane (object):
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

def plane_intersect(ray, plane):
    if not isinstance(ray, Ray):
        raise TypeError('ray must be a Ray')
    if not isinstance(plane, Plane):
        raise TypeError('plane must be a Plane')
    numer = (p0 - ray.origin) @ normal
    denom = ray.dir @ normal
    if numer == 0 or denom == 0: return np.NaN
    ret = numer/denom
    return ret if ret > 0 else np.NaN

def sphere_intersect(ray, sphere):
    if not isinstance(ray, Ray):
        raise TypeError('ray must be a Ray')
    if not isinstance(sphere, Sphere):
        raise TypeError('sphere must be a Sphere')
    OS = ray.origin - s0
    B = 2*ray.dir @ OS
    C = OS @ OS - radius**2
    disc = B**2 - 4*C
    if disc > 0:
        dist = np.sqrt(disc)
        Q = (-B - dist)/2 if B < 0 else (-B + dist)/2
        ret0, ret1 = sorted([Q, C/Q])
        if not ret1 < 0: return ret1 if ret0 < 0 else ret0
    return np.NaN
