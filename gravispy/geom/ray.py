import numpy as np

class Ray (object):
    def __init__(self, origin, direction):
        if not isinstance(origin, (list, tuple, np.ndarray)) or len(origin) not in (2,3):
            raise TypeError('origin must be either 2D or 3D cartesian coordinates')
        if not isinstance(direction, (list, tuple, np.ndarray)) or len(direction) not in (2,3):
            raise TypeError('direction must be given as either a pair of angles or a 3D vector')

        if len(origin) is 2: origin = list(origin) + [0]
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

    def plane_intersect(self, *args):
        if len(args) is not 2:
            raise TypeError('plane required as argument defined by an origin and normal')
        p0 = np.array(args[0])
        normal = np.array(args[1])
        numer = (p0 - self.origin).dot(normal)
        denom = self.dir.dot(normal)
        if numer == 0 or denom == 0: return np.Infinity
        else: return numer/denom

    def __str__(self):
        return '{}(O:{}, D:{})'.format(self.__class__.__name__, self.origin, self.dir)

    def __repr__(self):
        return str(self.__class__)
