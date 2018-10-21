import numpy as np
import itertools as it
from sympy import *

class Metric (object):
    def __init__(self, coords, matrix, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs

        self.is_metric = True
        self.is_spacetime = False
        self._A = Matrix(matrix)

        if not self._A.is_square: raise ValueError('Matrix must be square.')
        if not all(map(isinstance, coords, len(coords)*[Symbol])):
            raise TypeError('coordinates must be Symbols')

        self.shape = self._A.shape
        self._coords = tuple(coords)
        self._basis = self._coords
        self._vars = tuple(v for v in self.__args if isinstance(v, Symbol))
        self.coords = dict(zip(map(str, self._coords), self._coords))
        self.vars = dict(zip(map(str, self._vars), self._vars))
        self.conditions = {}
        self.assumptions = {
                'spherical' : False,
                'static' : False,
                }

        self._T = None
        self._I = None
        self._Agenerator = None
        self._Tgenerator = None
        self._Igenerator = None

        self._set_generators()

    def _set_generators(self):
        self._T = self._A.T
        self._I = self._A.inv(method=self.__kwargs.get('inv_method', None))
        self._Agenerator = lambdify(self._coords + self._vars,
                                    self._A,
                                    modules=self.__kwargs.get('lambdify_modules', None))
        self._Tgenerator = lambdify(self._coords + self._vars,
                                    self._T,
                                    modules=self.__kwargs.get('lambdify_modules', None))
        self._Igenerator = lambdify(self._coords + self._vars,
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
        else: return Metric(self._coords, res, *self.__args, **self.__kwargs)

    def inv(self):
        return Metric(self._coords, self._I, *self.__args, **self.__kwargs)

    def transpose(self):
        return Metric(self._coords, self._T, *self.__args, **self.__kwargs)

    def set_conditions(self, *args):
        if (not all(map(isinstance, args, len(args)*[tuple]))
                or any([len(arg) is not 2 for arg in args])):
            raise TypeError('arguments must be tuples of length 2')

        sub_dict = dict(args)
        self._A = self._A.subs(sub_dict)
        self.conditions.update(sub_dict)

        coords_selector = [coord in sub_dict.keys() for coord in self._coords]
        vars_selector = [var in sub_dict.keys() for var in self._vars]
        coord_keys = tuple(it.compress(self._coords, coords_selector))
        var_keys = tuple(it.compress(self._vars, vars_selector))

        self.coords.update(dict(zip(map(str, coord_keys), [sub_dict[key] for key in coord_keys])))
        self.vars.update(dict(zip(map(str, var_keys), [sub_dict[key] for key in var_keys])))

        for idx in range(len(self._coords)):
            if coords_selector[idx]:
                self._A.row_del(idx)
                self._A.col_del(idx)

        for idx in range(self._A.rows):
            if self._A.row(idx).is_zero and self._A.col(idx).is_zero:
                self._A.row_del(idx)
                self._A.col_del(idx)

        self._coords = tuple(it.compress(self._coords, np.logical_not(coords_selector)))
        self._vars = tuple(it.compress(self._vars, np.logical_not(vars_selector)))

        self._set_generators()

    def __getitem__(self, key):
        elem = np.asarray(self._A).__getitem__(key)
        return lambdify(self._coords + self._vars, elem, modules=self.__kwargs.get('lambdify_modules', None))

    def __call__(self, *args):
        return self.A(*args)

    def __str__(self):
        return self.as_ndarray().__str__()

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.as_ndarray().__str__())

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
        self.signature = tuple(self.signature)

        super(Minkowski, self).__init__([], diag(*self.signature), **kwargs)

class Schwarzschild (Metric):
    def __init__(self, coords, mass, timelike=True, **kwargs):
        self.__kwargs = kwargs

        self.signature = ones(1,4)
        self.signature[0] *= -1
        if timelike: self.signature *= -1
        self.signature = tuple(self.signature)

        if len(coords) is not 4: raise ValueError('Invalid number of coordinates')
        t, r, th, ph = coords
        gamma = 1 - 2*mass/r

        super(Schwarzschild, self).__init__(
                coords, diag(gamma, 1/gamma, r**2, r**2*sin(th)**2) * diag(*self.signature),
                mass, **kwargs)

        self.is_spacetime = True
        self.is_timelike = timelike

        self.assumptions['spherical'] = True
        self.assumptions['static'] = True

    def _set_generators(self):
        super(Schwarzschild, self)._set_generators()

        self._conformal_factor = None
        self._radial_factor = None
        self._angular_factor = None
        self._cfgenerator = None
        self._rfgenerator = None
        self._afgenerator = None

    def conformal_factor(self, *args):
        if not self._conformal_factor:
            self._conformal_factor = self.signature[0]*self._A[0,0]
            self._cfgenerator = lambdify(self._coords + self._vars,
                                         self._conformal_factor,
                                         modules=self.__kwargs.get('lambdify_modules', None))
        if not self._conformal_factor.free_symbols or args:
            return self._cfgenerator(*args)
        else:
            return self._conformal_factor

    def radial_factor(self, *args):
        if not self._radial_factor:
            if self._radial_factor is 0: return 0
            self._radial_factor = self.signature[1]*simplify(self._A[1,1]/self.conformal_factor())
            self._rfgenerator = lambdify(self._coords + self._vars,
                                         self._radial_factor,
                                         modules=self.__kwargs.get('lambdify_modules', None))
        if not self._radial_factor.free_symbols or args:
            return self._rfgenerator(*args)
        else:
            return self._radial_factor

    def angular_factor(self, *args):
        if not self._angular_factor:
            if self._angular_factor is 0: return 0
            self._angular_factor = self.signature[2]*simplify(self._A[2,2]/self.conformal_factor())
            self._afgenerator = lambdify(self._coords + self._vars,
                                         self._angular_factor,
                                         modules=self.__kwargs.get('lambdify_modules', None))
        if not self._angular_factor.free_symbols or args:
            return self._afgenerator(*args)
        else:
            return self._angular_factor

    def set_conditions(self, *args):
        super(Schwarzschild, self).set_conditions(*args)

        new_signature = list(self.signature)
        for idx,coord in enumerate(self._basis):
            if coord not in self._coords: new_signature.pop(idx)
        self.signature = tuple(new_signature)

        if self._radial_factor is not 0 and self._basis[1] not in self._coords:
            self._radial_factor = 0
        if self._angular_factor is not 0 and self._basis[2] not in self._coords:
            self._angular_factor = 0
