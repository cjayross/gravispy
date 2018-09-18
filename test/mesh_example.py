import numpy as np
from vispy import app, gloo, io
from vispy.util.transforms import perspective, rotate, translate

VSHADER = """
#version 120

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
attribute vec3 a_position;
attribute vec3 a_normal;
varying vec3 v_position;

void main ( void )
{
    v_position = a_position;
    // center the teapot by the average of a_position.xy
    vec3 position = a_position - vec3( .05393738, 1.724137, 0. );
    gl_Position = u_proj * u_view * u_model * vec4( position, 1. );
}
"""

FSHADER = """
#version 120

varying vec3 v_position;

void main ( void )
{
    gl_FragColor = vec4( v_position, 1. );
}
"""

def teapot ():
    vtype = [('a_position', np.float32, 3),
             ('a_normal', np.float32, 3)]
    V, F, N, _ = io.read_mesh('./obj/teapot.obj')
    vertices = np.zeros( len(V), dtype=vtype )
    vertices['a_position'] = V
    vertices['a_normal'] = N
    return vertices, F

class Canvas ( app.Canvas ):

    def __init__ ( self ):
        app.Canvas.__init__( self, keys='interactive', size=(800,600) )

        self.V, self.F = teapot()

        self.program = gloo.Program( VSHADER, FSHADER )
        self.program.bind(gloo.VertexBuffer(self.V))
        self.FF = gloo.IndexBuffer(self.F)

        self.program['u_model'] = np.eye( 4, dtype=np.float32 )
        self.program['u_view'] = translate((0,0,-10))
        self.theta, self.phi = 0, 0

        self.set_viewport()

        self.timer = app.Timer( 'auto', self.on_timer, start=True )

        gloo.set_state(depth_test=True)
        self.show()

    def on_draw ( self, event ):
        gloo.clear(color='black', depth=True)
        self.program.draw('triangles', self.FF)

    def on_resize ( self, event ):
        self.set_viewport()

    def set_viewport ( self ):
        width, height = self.size
        self.program['u_proj'] = perspective( 45., width/float(height), 2., 20. )
        gloo.set_viewport( 0, 0, *self.physical_size )

    def on_timer ( self, event ):
        self.theta += .5
        self.phi += .5
        self.program['u_model'] = np.dot( rotate(self.theta, (0,0,1)),
                                          rotate(self.phi, (0,1,0) ))
        self.update()

if __name__ == '__main__':
    canvas = Canvas()
    app.run()
