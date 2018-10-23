import numpy as np

class Ray (object):
    def __init__(self, origin, direction):
        if not isinstance(origin, (list, tuple, np.ndarray)) or len(origin) is not 3:
            raise TypeError('origin must be 3D cartesian coordinates')
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

def plane_intersect(ray, p0, normal):
    if not isinstance(ray, Ray):
        raise TypeError('ray must be a Ray object')
    numer = (p0 - ray.origin) @ normal
    denom = ray.dir @ normal
    if numer == 0 or denom == 0: return np.NaN
    ret = numer/denom
    return ret if ret > 0 else np.NaN

def sphere_intersect(ray, s0, radius):
    if not isinstance(ray, Ray):
        raise TypeError('ray must be a Ray object')
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
