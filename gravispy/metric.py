from ..ext.gravipy import *

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
        self._coords = { 't' : self.coords(-1),\
                         'r' : self.coords(-2),\
                         'theta' : self.coords(-3),\
                         'phi' : self.coords(-4) }
        self._affine_parameter = symbols('eta', real=True)
        self._principal = None
        self._principal_parameters = None
        self.mass = mass

    def has_principal (self): return self._principal is not None

    def set_principal_function (self, radial_part, angular_part, mass, energy, angular_mom ):
        """
        Sets the Hamiltonian for the Schwarzschild metric.

        Expressed as
        S(\\eta, t, r, \\theta, \\phi) = \\frac{m^2}{2}\\eta - Et + L\\phi + F(r) + G(\\theta)

        where E is the energy of the particle, L the angular momentum, and m the mass.
        F is referred to as the radial dependence of the Hamiltonian, and G is referred to as the angular dependence.

        Parameters
        ----------
        radial_part : Function
            The unknown radial dependence.
        angular_part : Function
            The unknown angular dependence.
        mass : Symbol | float
            The mass of the particle being described. Set to 0 for null paths.
        energy : Symbol | float
            The energy of the particle.
        angular_mom : Symbol | float
            The angular momentum of the particle.
        """
        self._principal = mass**2*self._affine_parameter/2\
                        - energy*self._coords['t']\
                        + angular_mom*self._coords['phi']\
                        + radial_part(self._coords['r'])\
                        + angular_part(self._coords['theta'])
        self._principal_parameters = { 'radial' : radial_part(self._coords['r']),\
                                       'angular' : angular_part(self._coords['theta']) }
        return self._principal

    def geodesic_equations (self):
        """
        Symbolic representation of the geodesic (orbital) equations of a particle.

        Expressed as the right-hand side of the equations
        \\frac{\\partial x^\\mu}{\\partial \\eta} = -g^{\\mu\\nu}\\frac{\\partial S}{\\partial x^\\nu}

        Note
        ----
        Given the expressions returned by this function; one can convert them into a typical, callable function by
        >>> from sympy import lambdify
        >>> geodesics = g.geodesic_equations()
        >>> lambdified_geodesic = lambdify(g._coords['t'], geodesics[0], module='numpy')
        >>> lambdified_geodesic(0.233)
        8.71236

        The above example assumes that 't' is the only symbolic term in geodesics[0].
        Additional symbols can be specified by passing a tuple as the first argument the lambdify.
        """
        if not self.has_principal():
            raise ValueError('geodesics require the principal function to be set')
        S = self._principal
        gradS = Matrix([ S.diff(x) for x in self.coords.c ]) # note: self.coords.c is a list of coordinate symbols
        return simplify(-self.metric(-All,-All) * gradS)

    def hamilton_jacobi (self, constrained=True):
        """
        Symbolic representation of the Hamilton-Jacobi equation for the Schwarzschild metric.

        Expressed as the left-hand side of the equations
        \\frac{\\partial S}{\\partial\\eta} - \\frac{1}{2}g^{\\mu\\nu}\\frac{\\partial S}{\\partial x^\\mu}\\frac{\\partial S}{\\partial x^\\nu} = 0

        Parameters
        ----------
        constrained : bool
            Specifies whether or not to apply the equitorial constraint on the parameters theta and G(theta).
            This will set theta = pi / 2 and diff(G(theta), theta) = 0.
        """
        if not self.has_principal():
            raise ValueError('hamilton_jacobi requires the principal function to be set')
        S = self._principal
        gradS = Matrix([ S.diff(x) for x in self.coords.c ])
        HJ = S.diff(self._affine_parameter)\
             - Rational(1,2) * (gradS.T * self.metric(-All,-All) * gradS)[0]
        HJ = simplify(HJ)
        if not constrained: return HJ
        else: return HJ.subs({ diff(self._principal_parameters['angular'], self._coords['theta']) : 0,\
                               self._coords['theta'] : pi / 2 })
