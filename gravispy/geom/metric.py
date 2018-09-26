from ..ext.gravipy import All, MetricTensor, Matrix, Rational, simplify, symbols, diff, diag, sin, pi

class Schwarzschild (MetricTensor):
    """
    The Schwarzschild metric.

    The metric is defined (in the default timelike signature):
    Matrix([
    [1 - 2*M/r,              0,     0,                   0],
    [        0, -1/(1 - 2*M/r),     0,                   0],
    [        0,              0, -r**2,                   0],
    [        0,              0,     0, -r**2*sin(theta)**2]])

    Parameters
    ----------
    symbol : str
        The symbol used to represent the metric.
    coords : Coordinates
        Coordinates as defined in ext.gravipy
    mass : Symbol | float
        Mass of the reference object; provided either symbolically with sympy or literally
    """

    def __init__ (self, symbol, coords, mass, timelike=True):
        metric = diag( 1 - 2*mass/coords(-2),
                       -1 / (1 - 2*mass/coords(-2)),
                       -coords(-2)**2,
                       -coords(-2)**2*sin(coords(-3))**2 )
        if not timelike: metric = -1 * metric
        super(Schwarzschild, self).__init__(symbol, coords, metric)

        self.is_timelike = timelike
        self.is_spacelike = not timelike
        self._affine_parameter = symbols('eta', real=True)
        self._principal = None
        self._principal_parameters = None
        self.mass = mass

    def has_principal (self): return self._principal is not None

    def set_principal_function (self, radial_part, angular_part, params):
        self._principal = params[0]**2*self._affine_parameter/2\
                        - params[1]*self.coords(-1)\
                        + params[2]*self.coords(-4)\
                        + radial_part(self.coords(-2))\
                        + angular_part(self.coords(-3))
        self._principal_parameters = [ radial_part(self.coords(-2)), angular_part(self.coords(-3)) ]
        return self._principal

    def geodesic_functions (self):
        if not self.has_principal():
            raise ValueError('geodesics require the principal function to be set')
        S = self._principal
        gradS = Matrix([ S.diff(x) for x in self.coords(-All)[:,0] ])
        return simplify(-self.metric(-All,-All) * gradS)

    def hamilton_jacobi (self, constrained=True):
        if not self.has_principal():
            raise ValueError('hamilton_jacobi requires the principal function to be set')
        S = self._principal
        gradS = Matrix([ S.diff(x) for x in self.coords(-All)[:,0] ])
        HJ = S.diff(self._affine_parameter)\
             - Rational(1,2) * (gradS.T * self.metric(-All,-All) * gradS)[0]
        HJ = simplify(HJ)
        if not constrained: return HJ
        else: return HJ.subs({ diff(self._principal_parameters[1], self.coords(-3)) : 0,\
                               self.coords(-3) : pi / 2 })
