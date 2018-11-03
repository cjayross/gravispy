import numpy as np
import itertools as it
from sympy import Matrix, Symbol, lambdify, symbols, simplify,\
                  zeros, ones, eye, diag, sin

__all__ = [
        'Metric',
        'Euclidean',
        'SpacetimeMetric',
        'Minkowski',
        'SphericalSpacetime',
        'Schwarzschild',
        ]

class Metric (object):
    """
    Abstract metric object.
    Metrics are numpy arrays of callable functions representing each components
    of a metric tensor.

    Parameters
    ==========
    coords: iterable
        List of Symbols denoting the coordinates used for argument
        bookkeeping.
    matrix: Matrix | ndarray | list | tuple
        Matrix representing representing the components of the metric.
    *args: Symbols
        Symbols representing additional dependencies of the metric.

    Examples
    ========
    >>> from numpy import pi
    >>> from sympy import symbols, trace
    >>> from gravispy.geom.metric import Schwarzschild

    >>> t, r, th, ph, M = symbols('t r theta phi M', real=True)
    >>> S = Schwarzschild([t, r, th, ph], M, timelike=True)

    >>> S.as_Matrix()
    Matrix([
    [-2*M/r + 1,              0,     0,                   0],
    [         0,-1/(-2*M/r + 1),     0,                   0],
    [         0,              0, -r**2,                   0],
    [         0,              0,     0, -r**2*sin(theta)**2]])

    >>> S.args
    (r, theta, M)
    >>> S(10, pi/2, 1)
    array([[ 0.8 , 0.    , 0.    , 0.    ],
           [ 0.  , -1.25 , 0.    , 0.    ],
           [ 0.  , 0.    , -100. , 0.    ],
           [ 0.  , 0.    , 0.    , -100. ]])
    >>> S[0,0](10, pi/2, 1)
    0.8
    >>> S(10, pi/2, 1)[1,1]
    -1.25

    >>> S.set_conditions((M, 1))
    >>> S.as_Matrix()
    Matrix([
    [-2/r + 1,              0,     0,                   0],
    [         0,-1/(-2/r + 1),     0,                   0],
    [         0,            0, -r**2,                   0],
    [         0,            0,     0, -r**2*sin(theta)**2]])
    >>> S.args
    (r, theta)
    >>> S.vars
    {'M': 1}

    >>> S.set_conditions((th, pi/2))
    >>> S.as_Matrix()
    Matrix([
    [-2/r + 1,              0, 0,         0],
    [         0,-1/(-2/r + 1), 0,         0],
    [         0,            0, 0,         0],
    [         0,            0, 0, -1.0*r**2]])
    >>> S.args
    (r,)
    >>> S.coords
    {'t': t, 'r': r, 'theta': 1.5707963267948966, 'phi': phi}

    >>> S.conditions
    {'M': 1, 'theta': 1.5707963267948966}
    >>> S(20) # only requires one argument under these conditions.
    array([[ 0.9 , 0.         , 0. , 0.    ],
           [ 0.  , -1.11111111, 0. , 0.    ],
           [ 0.  , 0.         , 0. , 0.    ],
           [ 0.  , 0.         , 0. , -400. ]])

    >>> S.applyfunc(trace)
    -1.0*r**2 + 1 - 1/(1 - 2/r) - 2/r

    >>> Si = S.inv()
    >>> Si.as_Matrix()
    Matrix([
    [1/(2/r + 1),       0, 0,         0],
    [          0,-1 + 2/r, 0,         0],
    [          0,       0, 0,         0],
    [          0,       0, 0, -1.0/r**2]])
    >>> Si.conditions
    {'M': 1, 'theta': 1.5707963267948966}
    """
    def __init__(self, coords, matrix, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs
        self._inv_method = self.__kwargs.get('inv_method', None)
        self._lambdify_modules = self.__kwargs.get('lambdify_modules', None)

        self.is_metric = True
        self.is_spacetime = False
        self._A = Matrix(matrix)

        if not self._A.is_square:
            raise ValueError('Matrix must be square.')
        if not all(map(isinstance, coords, len(coords)*[Symbol])):
            raise TypeError('coordinates must be Symbols')

        self.shape = self._A.shape
        self.dims = self.shape[0]
        # the basis remains constant
        self.basis = tuple(coords)
        # coordinates are used as arguments to lambdified metric compononents.
        # self._coords is subject to change depending on conditions
        # set by the user.
        self._coords = self.basis
        # variables are used as arguments to lambdified metric compononents.
        # self._vars is subject to change depending on conditions
        # set by the user.
        self._vars = tuple(v for v in self.__args if isinstance(v, Symbol))
        # self.coords and self.vars are the canonical representation for the
        # metric arguments. they retain their initial values as keys to the
        # dictionary but their values are subject to change based on conditions
        # set by the user.
        self.coords = dict(zip(map(str, self._coords), self._coords))
        self.vars = dict(zip(map(str, self._vars), self._vars))
        # self.args is the tuple representing what will be used in lambdified
        # metric components. their order is to remain constant but elements may
        # be removed by conditions.
        coord_args = (x for x in self._coords if x in self._A.free_symbols)
        var_args = (v for v in self._vars if v in self._A.free_symbols)
        self.args = tuple(coord_args) + tuple(var_args)

        if len(self.basis) is not self.shape[0]:
            raise ValueError('coordinates do not match metric dimensions')
        elif len(self.args) < len(self._A.free_symbols):
            raise ValueError('coordinates and variables given do not '\
                             'sufficiently describe the metric')

        self.conditions = self.__kwargs.get('conditions', {})
        self.assumptions = {
                'spherical' : False,
                'axial' : False,
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
        # the reduced metric is a nonsingular version of the metric.
        # singular metrics may arise from conditions set by the user.
        reduced_metric = self._A.copy()
        zero_idxs = [idx for idx in range(reduced_metric.rows)
                     if (reduced_metric.row(idx).is_zero
                         and reduced_metric.col(idx).is_zero)]
        for offset, idx in enumerate(zero_idxs):
            reduced_metric.row_del(idx-offset)
            reduced_metric.col_del(idx-offset)
        self._I = reduced_metric.inv(method=self._inv_method)
        for idx in zero_idxs:
            # replace the missing row and column.
            self._I = self._I.col_insert(idx, zeros(self._I.shape[0], 1))\
                             .row_insert(idx, zeros(1, self._I.shape[1]+1))
        self._Agenerator = lambdify(self.args, self._A,
                                    modules=self._lambdify_modules)
        self._Tgenerator = lambdify(self.args, self._T,
                                    modules=self._lambdify_modules)
        self._Igenerator = lambdify(self.args, self._I,
                                    modules=self._lambdify_modules)

    def A(self, *args):
        return self._Agenerator(*args)

    def T(self, *args):
        return self._Tgenerator(*args)

    def I(self, *args):
        return self._Igenerator(*args)

    def as_Matrix(self):
        return self._A.copy()

    def as_ndarray(self):
        return np.asarray(self._A)

    def applyfunc(self, func=Matrix.applyfunc, return_metric=False,
                  *args, **kwargs):
        res = self._A.__getattribute__(func.__name__)(*args, **kwargs)
        if not isinstance(res, Matrix) or not return_metric:
            return res
        else:
            return Metric(self.basis, res, conditions=self.conditions,
                          *self.__args, **self.__kwargs)

    def inv(self):
        return Metric(self.basis, self._I, conditions=self.conditions,
                      *self.__args, **self.__kwargs)

    def transpose(self):
        return Metric(self.basis, self._T, conditions=self.conditions,
                      *self.__args, **self.__kwargs)

    def set_conditions(self, *args):
        if (not all(map(isinstance, args, len(args)*[tuple]))
                or any([len(arg) is not 2 for arg in args])):
            raise TypeError('arguments must be tuples of length 2')
        if not all(map(
                lambda a: isinstance(a[1],(int,float)) or a[1].is_number,
                args)):
            raise ValueError('conditionals must be constants')

        sub_dict = dict(args)
        self._A = self._A.subs(sub_dict)
        self.conditions.update(
                zip(map(str, sub_dict.keys()), sub_dict.values()))

        coords_selector = [coord in sub_dict.keys() for coord in self._coords]
        vars_selector = [var in sub_dict.keys() for var in self._vars]
        coord_keys = tuple(it.compress(self._coords, coords_selector))
        var_keys = tuple(it.compress(self._vars, vars_selector))

        for idx in range(len(self._coords)):
            if coords_selector[idx]:
                # replace the metric column and row of the coordinate
                # with 0 since the coordinate's differential is 0
                self._A.row_del(idx)
                self._A.col_del(idx)
                self._A = self._A.col_insert(idx, zeros(self._A.shape[0], 1))\
                                 .row_insert(idx, zeros(1, self._A.shape[1]+1))

        self.coords.update(dict(zip(map(str, coord_keys),
                                    [sub_dict[key] for key in coord_keys])))
        self.vars.update(dict(zip(map(str, var_keys),
                                  [sub_dict[key] for key in var_keys])))

        self._coords = tuple(it.compress(self._coords,
                                         np.logical_not(coords_selector)))
        self._vars = tuple(it.compress(self._vars,
                                       np.logical_not(vars_selector)))
        # update the args such that they preserve their ordering
        coord_args = (x for x in self._coords if x in self._A.free_symbols)
        var_args = (v for v in self._vars if v in self._A.free_symbols)
        self.args = tuple(coord_args) + tuple(var_args)

        self._set_generators()

    def __getitem__(self, key):
        elem = np.asarray(self._A).__getitem__(key)
        return lambdify(self.args, elem, modules=self._lambdify_modules)

    def __call__(self, *args):
        return self.A(*args)

    def __str__(self):
        return self.as_ndarray().__str__()

    def __repr__(self):
        return str(self.__class__)

class Euclidean (Metric):
    def __init__(self, ndim, **kwargs):
        super(Euclidean, self).__init__([], Matrix.eye(ndim), **kwargs)

class SpacetimeMetric (Metric):
    def __init__(self, coords, matrix, *args, **kwargs):
        self.signature = ones(1,4)
        self.signature[0] *= -1
        if kwargs.get('timelike', True):
            self.signature *= -1
        self.signature = tuple(self.signature)
        matrix = Matrix(matrix)
        # note: empty Matrix is considered square
        if not matrix.is_square:
            raise ValueError('spacetime metrics must be defined '\
                             'by square matrices')
        elif matrix.shape[0] is 0:
            matrix = eye(4)
        elif matrix.shape[0] is not 4:
            raise ValueError('non-default spacetime metrics '\
                             'must have 4 dimensions')

        super(SpacetimeMetric, self).__init__(
                coords, matrix * diag(*self.signature), *args, **kwargs)

        self.is_spacetime = True
        self.is_timelike = kwargs.get('timelike', True)
        self.is_spacelike = not self.is_timelike

class Minkowski (SpacetimeMetric):
    def __init__(self, timelike=True, **kwargs):
        super(Minkowski, self).__init__([], Matrix(), **kwargs)

        self.assumptions['static'] = True

class SphericalSpacetime (SpacetimeMetric):
    def __init__(self, coords, matrix, *args, **kwargs):
        super(SphericalSpacetime, self).__init__(
                coords, matrix, *args, **kwargs)

        self.assumptions['spherical'] = True
        self.assumptions['axial'] = True
        self.assumptions['static'] = False

    def _set_generators(self):
        super(SphericalSpacetime, self)._set_generators()

        self._conformal_factor = None
        self._radial_factor = None
        self._angular_factor = None
        self._cfgenerator = None
        self._rfgenerator = None
        self._afgenerator = None

    def conformal_factor(self, *args):
        if not self._conformal_factor:
            if (len(self._A[0,0].free_symbols) > 1
                    or (len(self._A[0,0].free_symbols) is 1
                        and self.basis[1] not in self._A[0,0].free_symbols)):
                raise ValueError('metric is not spherically symmetric')
            self._conformal_factor = self.signature[0]*self._A[0,0]
            self._cfgenerator = lambdify(self.args, self._conformal_factor,
                                         modules=self._lambdify_modules)
        if args:
            return self._cfgenerator(*args)
        else:
            return self._conformal_factor

    def radial_factor(self, *args):
        if not self._radial_factor:
            if self._radial_factor == 0.: return 0.
            if (len(self._A[1,1].free_symbols) > 1
                    or (len(self._A[1,1].free_symbols) is 1
                        and self.basis[1] not in self._A[1,1].free_symbols)):
                raise ValueError('metric is not spherically symmetric')
            self._radial_factor =\
                    self.signature[1]\
                    * simplify(self._A[1,1]/self.conformal_factor())
            self._rfgenerator = lambdify(self.args, self._radial_factor,
                                         modules=self._lambdify_modules)
        if args:
            return self._rfgenerator(*args)
        else:
            return self._radial_factor

    def angular_factor(self, *args):
        if not self._angular_factor:
            if self._angular_factor == 0.: return 0.
            if (len(self._A[2,2].free_symbols) > 1
                    or (len(self._A[2,2].free_symbols) is 1
                        and self.basis[1] not in self._A[2,2].free_symbols)):
                raise ValueError('metric is not spherically symmetric')
            self._angular_factor =\
                    self.signature[2]\
                    * simplify(self._A[2,2]/self.conformal_factor())
            self._afgenerator = lambdify(self.args, self._angular_factor,
                                         modules=self._lambdify_modules)
        if args:
            return self._afgenerator(*args)
        else:
            return self._angular_factor

    def set_conditions(self, *args):
        super(SphericalSpacetime, self).set_conditions(*args)

        if self.basis[1] not in self._coords:
            self._radial_factor = self._radial_factor.subs(
                    {self.basis[1]: self.coords[self.basis[1]]})
            self._angular_factor = self._angular_factor.subs(
                    {self.basis[1]: self.coords[self.basis[1]]})
            self._conformal_factor = self._conformal_factor.subs(
                    {self.basis[1]: self.coords[self.basis[1]]})

class Schwarzschild (SphericalSpacetime):
    """
    The Schwarzschild metric.

    The metric is defined (in the default timelike signature):
    Matrix([
    [1 - 2*M/r,              0,     0,                   0],
    [        0, -1/(1 - 2*M/r),     0,                   0],
    [        0,              0, -r**2,                   0],
    [        0,              0,     0, -r**2*sin(theta)**2]])

    Parameters
    ==========
    mass : Symbol | float | int
        Mass of the reference object; provided either symbolically with sympy
        or literally as an integer or float.
        Note that if this is zero, then the metric reduces to the Minkowski
        metric for spherical coordinates.
    coords : iterable of Symbols
        Coordinate variables to use.
        Default coordinates are (t, r, theta, phi)
    """
    def __init__(self, mass=Symbol('M', real=True),
                 coords=symbols('t r theta phi', real=True),
                 timelike=True, **kwargs):
        self.__kwargs = kwargs
        self.mass = mass

        if len(coords) is not 4:
            raise ValueError('Invalid number of coordinates')
        t, r, th, ph = coords
        gamma = 1 - 2*self.mass/r

        super(Schwarzschild, self).__init__(
                coords, diag(gamma, 1/gamma, r**2, r**2*sin(th)**2),
                mass, **kwargs)

        self.assumptions['static'] = True

    def set_conditions(self, *args):
        super(Schwarzschild, self).set_conditions(*args)

        if (hasattr(self.mass, 'is_Symbol') and self.mass.is_Symbol
                and self.mass not in self._vars):
            self.mass = self.vars[str(self.mass)]

    @property
    def radius(self):
        return 2*self.mass
