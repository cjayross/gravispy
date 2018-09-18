from vispy import app, gloo
from numpy import sin

VSHADER = """
#version 120

attribute vec3 a_position;
varying vec2 v_position;

void main ( void )
{
    v_position = a_position.xy;
    gl_Position = vec4( a_position, 1. );
}
"""

FSHADER = """
#version 120

const float radius = 0.3;

uniform float u_aspect_ratio;
uniform vec3 u_color;
uniform vec2 u_center;
varying vec2 v_position;

void main ( void )
{
    vec2 pos = vec2( u_aspect_ratio * v_position.x, v_position.y );
    if ( distance(pos, u_center) < radius )
    { gl_FragColor = vec4( u_color, 1. ); }
    else
    { discard; }
}
"""

class Canvas ( app.Canvas ):
    def __init__ ( self ):
        app.Canvas.__init__( self, position=(300,100),
                             size=(800,600), keys='interactive' )

        self.circle1 = gloo.Program( VSHADER, FSHADER )
        self.circle2 = gloo.Program( VSHADER, FSHADER )

        self.circle1['u_color'] = (0.,0.,0.)
        self.circle2['u_color'] = (1.,1.,1.)
        self.circle1['u_center'] = (-.5, 0)
        self.circle2['u_center'] = (+.5, 0)
        # circle1 is set so that it is closer to the camera.
        self.circle1['a_position'] = [ (-1, -1, -.5), (-1, +1, -.5),
                                       (+1, -1, -.5), (+1, +1, -.5) ]
        self.circle2['a_position'] = [ (-1, -1, +.5), (-1, +1, +.5),
                                       (+1, -1, +.5), (+1, +1, +.5) ]

        # This setting calls for a depth test; without it, circle2 ( which is further from the screen than circle1 )
        # will be displayed in front of circle1 since it is the last to be drawn.
        # With a depth test, a pixel can only be overwritten if it's fragment position has a smaller z-value.
        gloo.set_state(depth_test=True)
        self.set_viewport()
        self.timer = app.Timer('auto', connect=self.on_timer, start=True)

        self.show()

    def on_resize ( self, event ):
        self.set_viewport()

    def set_viewport ( self ):
        width, height = self.size
        gloo.set_viewport( 0, 0, *self.physical_size )
        self.circle1['u_aspect_ratio'] = width/float(height)
        self.circle2['u_aspect_ratio'] = width/float(height)

    def on_timer ( self, event ):
        t = event.elapsed
        self.circle1['u_center'] = ( -.5 + sin(2*t) / 2, 0 )
        self.circle2['u_center'] = ( +.5 - sin(2*t) / 2, 0 )
        self.update()

    def on_draw ( self, event ):
        gloo.clear(color='grey', depth=True)
        self.circle1.draw('triangle_strip')
        self.circle2.draw('triangle_strip')

if __name__ == '__main__':
    canvas = Canvas()
    app.run()
