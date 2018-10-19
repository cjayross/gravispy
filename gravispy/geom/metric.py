import numpy as np
from sympy import *

class Metric (object):
    def __init__(self, coords, matrix, *args, **kwargs):
        self.is_metric = True
        self.is_spacetime = False
        self.coords = tuple(coords)

        if not matrix.is_square:
            pass

        self._modules = kwargs.get('lambdify_modules', None)
        self._inv_method = kwargs.get('inv_method', None)
        self._A = Matrix(matrix)
        self._T = self._A.T
        self._I = self._A.inv(method=self._inv_method)
        self._Agenerator = lambdify(self.coords, self._A, modules=self._modules)
        self._Tgenerator = lambdify(self.coords, self._T, modules=self._modules)
        self._Igenerator = lambdify(self.coords, self._I, modules=self._modules)

        self.shape = self._A.shape

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
        else: return Metric(self.coords, res, lambdify_modules=self._modules, inv_method=self._inv_method)

    def inv(self):
        return Metric(self.coords, self._I, lambdify_modules=self._modules, inv_method=self._inv_method)

    def transpose(self):
        return Metric(self.coords, self._T, lambdify_modules=self._modules, inv_method=self._inv_method)

    def __call__(self, *args):
        return self.A(*args)

    def __getitem__(self, key):
        elem = np.asarray(self._A).__getitem__(key)
        return lambdify(self.coords, elem, self._modules)

class Euclidean (Metric):
    def __init__(self, coords, *args, **kwargs):
        pass

class Schwarzschild (Metric):
    def __init__(self, coords, *args, **kwargs):
        pass
