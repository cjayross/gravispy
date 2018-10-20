import numpy as np
from sympy import *

class MetricError (ValueError):
    pass

class Metric (object):
    def __init__(self, coords, matrix, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs

        self.is_metric = True
        self.is_spacetime = False
        self._A = Matrix(matrix)

        if not self._A.is_square: raise MetricError('Matrix must be square.')

        self.shape = self._A.shape
        self.coords = tuple(coords)
        self.variables = tuple(v for v in self.__args if isinstance(v, Symbol))

        self._T = self._A.T
        self._I = self._A.inv(method=self.__kwargs.get('inv_method', None))
        self._Agenerator = lambdify(self.coords + self.variables,
                                    self._A,
                                    modules=self.__kwargs.get('lambdify_modules', None))
        self._Tgenerator = lambdify(self.coords + self.variables,
                                    self._T,
                                    modules=self.__kwargs.get('lambdify_modules', None))
        self._Igenerator = lambdify(self.coords + self.variables,
                                    self._I,
                                    modules=self.__kwargs.get('lambdify_modules', None))

    def A(self, *args):
        return self._Agenerator(*args)

    def T(self, *args):
        return self._Tgenerator(*args)

    def I(self, *args):
        return self._Igenerator(*args)

    def as_Matrix(self):
        return self._A

    def as_ndarray(self):
        return np.asarray(self._A)

    def applyfunc(self, func=Matrix.applyfunc, return_metric=False, *args, **kwargs):
        res = self._A.__getattribute__(func.__name__)(*args, **kwargs)
        if not isinstance(res, Matrix) or not return_metric: return res
        else: return Metric(self.coords, res, *self.__args, **self.__kwargs)

    def inv(self):
        return Metric(self.coords, self._I, *self.__args, **self.__kwargs)

    def transpose(self):
        return Metric(self.coords, self._T, *self.__args, **self.__kwargs)

    def __call__(self, *args):
        return self.A(*args)

    def __getitem__(self, key):
        elem = np.asarray(self._A).__getitem__(key)
        return lambdify(self.coords, elem, modules=self.__kwargs.get('lambdify_modules', None))

class Euclidean (Metric):
    def __init__(self, ndim, **kwargs):
        super(Euclidean, self).__init__([], Matrix.eye(ndim), **kwargs)

class Minkowski (Metric):
    def __init__(self, timelike=True, **kwargs):
        self.is_spacetime = True
        self.is_timelike = timelike
        self.is_spacelike = not timelike

        self.signature = ones(1,4)
        self.signature[0] *= -1
        if timelike: self.signature *= -1
        super(Minkowski, self).__init__([], diag(*self.signature), **kwargs)

class Schwarzschild (Metric):
    def __init__(self, coords, mass, timelike=True, **kwargs):
        self.is_spacetime = True
        self.is_timelike = timelike
        self.is_spacelike = not timelike

        self.signature = ones(1,4)
        self.signature[0] *= -1
        if timelike: self.signature *= -1

        if len(coords) is not 4: raise MetricError('Invalid number of coordinates')
        t, r, th, ph = coords
        gamma = 1 - 2*mass/r

        super(Schwarzschild, self).__init__(
                coords, diag(gamma, 1/gamma, r**2, r**2*sin(th)**2) * diag(*self.signature),
                mass, **kwargs)
