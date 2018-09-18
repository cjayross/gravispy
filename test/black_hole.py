import numpy as np
from vispy import app, gloo

VSHADER = """
#version 120

// variables passed in from Canvas
attribute vec2 a_position;
attribute vec2 a_texcoord;
// variable that will be passed to fragment shader
varying vec2 v_position;

void main ( void )
{
    v_position = a_position;
    gl_Position = vec4( a_position, 0., 1. );
}

"""

FSHADER = """
#version 120

// variables set by Canvas that are constant through shader's execution
uniform float u_radius;
uniform float u_aspect_ratio;
// interpolated position recieved from vertex shader
varying vec2 v_position;

void main ( void )
{
    vec2 pos = vec2( u_aspect_ratio * v_position.x, v_position.y );
    if ( length(pos) < u_radius )
    {
        gl_FragColor = vec4( 0., 0., 0., 1. );
    }
    else
    {
        // define position dependent color gradient
        gl_FragColor = vec4( (pos.x + 1.) / 2., (pos.y + 1.) / 2., .5, 1. );
    }
}
"""

class Canvas ( app.Canvas ):
    def __init__ ( self ):
        # initialize the Canvas
        app.Canvas.__init__( self, position=(300,100),
                             size=(800,600), keys='interactive' )
        # compile the programs for the blackhole and backdrop
        self.program = gloo.Program( VSHADER, FSHADER )

        # pass the values that the shader requires
        self.program['a_position'] = [ (-1, -1), (-1, +1),
                                       (+1, -1), (+1, +1) ]
        self.program['u_radius'] = .3

        self.set_viewport()
        self.show()

    def on_resize ( self, event ):
        self.set_viewport()

    def set_viewport ( self ):
        # set aspect ratio
        width, height = self.size
        # this communicates to opengl the size of the window to render to
        gloo.set_viewport( 0, 0, *self.physical_size )
        # ensures that the black hole will be circular independently of the window's shape
        self.program['u_aspect_ratio'] = width/float(height)

    def on_draw ( self, event ):
        # draw the models by interpreting the 'a_position' list as the vertices of triangles
        self.program.draw('triangle_strip')

if __name__ == '__main__':
    canvas = Canvas()
    app.run()
