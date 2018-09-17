import numpy as np
from vispy import app, gloo

VSHADER_BLACKHOLE = """
#version 120

// attributes passed to shader from Canvas
attribute vec2 a_position;
// interpolated position between subsequent vertices in triangle strip
// passed on to fragment shader
varying vec2 v_position;

void main ( void )
{
    v_position = a_position;
    // add vertex to model
    gl_Position = vec4( a_position, 0., 1. );
}
"""

FSHADER_BLACKHOLE = """
#version 120

// values set from Canvas that don't change during this shader's execution
uniform float u_aspect_ratio;
uniform float u_radius;
// recieved position passed from vertex shader
varying vec2 v_position;

void main ( void )
{
    // scale the position appropriately
    vec2 position = vec2( u_aspect_ratio * v_position.x, v_position.y );
    if ( length(position) < u_radius )
    { gl_FragColor = vec4( 0., 0., 0., 1. ); }
    else
    // skip this pixel
    { discard; }
}
"""

VSHADER_BACKDROP = """
#version 120

attribute vec2 a_position;
attribute vec3 a_color;
varying vec3 v_color;

void main ( void )
{
    v_color = a_color;
    gl_Position = vec4( a_position, 1., 1. );
}
"""

FSHADER_BACKDROP = """
#version 120

varying vec3 v_color;

void main ( void )
{
    gl_FragColor = vec4( v_color, 1. );
}
"""

class Canvas ( app.Canvas ):
    def __init__ ( self ):
        # initialize the Canvas
        app.Canvas.__init__( self, position=(300,100),
                            size=(800,600), keys='interactive' )
        # compile the programs for the blackhole and backdrop
        self.blackhole = gloo.Program( VSHADER_BLACKHOLE,
                                       FSHADER_BLACKHOLE )
        self.backdrop = gloo.Program( VSHADER_BACKDROP,
                                      FSHADER_BACKDROP )

        # pass the values each shader requires
        self.backdrop['a_position'] = [ (-1, -1), (-1, +1),
                                        (+1, -1), (+1, +1) ]
        self.backdrop['a_color'] = [ (1, 0, 0), (0, 1, 0),
                                     (0, 0, 1), (1, 1, 0) ]

        self.blackhole['a_position'] = [ (-1, -1), (-1, +1),
                                         (+1, -1), (+1, +1) ]
        self.blackhole['u_radius'] = .3

        self.set_viewport()
        self.show()

    def on_resize ( self, event ):
        self.set_viewport()

    def set_viewport ( self ):
        # set aspect ratio
        width, height = self.size
        gloo.set_viewport( 0, 0, *self.physical_size )
        self.blackhole['u_aspect_ratio'] = width/float(height)

    def on_draw ( self, event ):
        # draw the models
        self.backdrop.draw('triangle_strip')
        self.blackhole.draw('triangle_strip')

if __name__ == '__main__':
    canvas = Canvas()
    app.run()
