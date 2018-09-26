from gravipy import All, MetricTensor, Matrix, Rational, simplify, symbols, diff, diag, sin, pi

class Schwarzschild ( MetricTensor ):

    def __init__ ( self, symbol, coords, mass ):
        metric = diag( 1 - 2*mass/coords(-2),
                       -1 / (1 - 2*mass/coords(-2)),
                       -coords(-2)**2,
                       -coords(-2)**2*sin(coords(-3))**2 )
        super( Schwarzschild, self ).__init__( symbol, coords, metric )

        self.is_timelike = True
        self.is_spacelike = False
        self._affine_parameter = symbols('eta', real=True)
        self._principal = None
        self._principal_parameters = None
        self.mass = mass

    def has_principal ( self ): return self._principal is not None

    def set_principal_function ( self, radial_part, angular_part, params ):
        self._principal = params[0]**2*self._affine_parameter/2\
                        - params[1]*self.coords(-1)\
                        + params[2]*self.coords(-4)\
                        + radial_part(self.coords(-2))\
                        + angular_part(self.coords(-3))
        self._principal_parameters = [ radial_part(self.coords(-2)), angular_part(self.coords(-3)) ]
        return self._principal

    def geodesic_functions ( self ):
        if not self.has_principal():
            raise ValueError('geodesics require the principal function to be set')
        S = self._principal
        gradS = Matrix([ S.diff(x) for x in self.coords(-All)[:,0] ])
        return simplify(-self.metric(-All,-All) * gradS)

    def hamilton_jacobi ( self, constrained=True ):
        if not self.has_principal():
            raise ValueError('hamilton_jacobi requires the principal function to be set')
        S = self._principal
        gradS = Matrix([ S.diff(x) for x in self.coords(-All)[:,0] ])
        HJ = S.diff(self._affine_parameter)\
             - Rational(1,2) * (gradS.T * self.metric(-All,-All) * gradS)[0]
        HJ = simplify(HJ)
        if not constrained: return HJ
        else: return HJ.subs({ diff( self._principal_parameters[1], self.coords(-3) ) : 0,\
                               self.coords(-3) : pi / 2 })
