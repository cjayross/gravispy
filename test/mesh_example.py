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
varying vec3 v_normal;

void main ( void )
{
    // center the teapot by the average of a_position.xy
    vec3 position = a_position - vec3( .05393738, 1.724137, 0. );
    gl_Position = u_proj * u_view * u_model * vec4( position, 1. );
    v_position = gl_Position.xyz;
    v_normal = (u_model * vec4( a_normal, 1. )).xyz;
}
"""

FSHADER = """
#version 120
const float M_PI = 3.14159265358979323846;
const float INFINITY = 1000000000.;

uniform vec3 u_O; //origin

uniform float u_ambient;
uniform float u_reflection;
uniform float u_light_intensity;
uniform vec3 u_light_position;
uniform vec3 u_light_color;
uniform vec2 u_light_specular;

uniform vec3 u_base_color;

varying vec3 v_position;
varying vec3 v_normal;

vec3 run( vec3 Q )
{
    vec3 N = normalize(v_normal);
    vec3 toO = normalize(u_O - Q);
    vec3 toL = normalize(u_light_position - Q);
    vec3 col_ray = vec3( u_ambient );

    col_ray += u_light_intensity * max( dot(N, toL), 0. ) * u_base_color;
    col_ray += u_light_specular.x * u_light_color
               * pow( max( dot( N, normalize(toL + toO) ), 0. ), u_light_specular.y );

    return clamp( u_reflection * col_ray, 0., 1. );
}

void main ( void )
{
    gl_FragColor = vec4( run(v_position), 1. );
}
"""

VSHADER_NORM = """
#version 120

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

attribute vec3 a_position;

void main ( void )
{
    vec3 position = a_position - vec3( .05393738, 1.724137, 0. );
    gl_Position = u_proj * u_view * u_model * vec4( position, 1. );
}
"""

FSHADER_NORM = """
#version 120

uniform vec3 u_color;

void main ( void )
{
    gl_FragColor = vec4( u_color, 1. );
}
"""

def teapot ():
    # this data type will keep the data recieved from the obj file organized.
    # vispy will automatically separate out the attributes included here and
    # set the vertex shader accordingly.
    vtype = [('a_position', np.float32, 3),
             ('a_normal', np.float32, 3)]
    # the last return value will be None since this object does not have any texture coordinates.
    V, F, N, _ = io.read_mesh('./obj/teapot.obj')
    vertices = np.zeros( len(V), dtype=vtype )
    vertices['a_position'] = V
    vertices['a_normal'] = N
    return vertices, F

class Canvas ( app.Canvas ):

    def __init__ ( self ):
        app.Canvas.__init__( self, keys='interactive', size=(800,600) )

        self.V, self.F = teapot()

        self.teapot = gloo.Program( VSHADER, FSHADER )
        # initialize a buffer of all the vertices
        self.teapot.bind(gloo.VertexBuffer(self.V))
        # each element in F is an array of 3 indices corresponding to the
        # locations within the vertex buffer that represent the face's 3 vertices.
        # when we draw the object, openGL will handle this association process automatically.
        self.FF = gloo.IndexBuffer(self.F)

        self.normals = gloo.Program( VSHADER_NORM, FSHADER_NORM )
        self.normals['a_position'] = np.array([*zip(self.V['a_position'],
                                                    self.V['a_position'] + self.V['a_normal'])])\
                                    .reshape(( 2 * len(self.V['a_normal']), 3 ))
        self.normals['u_color'] = (1,0,0)

        # u_view is the location of the camera.
        self.teapot['u_view'] = self.normals['u_view'] = translate((0,0,-10))
        # u_model is how the world is represented relative to the camera.
        # in this case, we intend on having the teapot spin around.
        self.teapot['u_model'] = self.normals['u_model'] = np.eye( 4, dtype=np.float32 )
        self.teapot['u_base_color'] = (0, 1, 0)
        self.teapot['u_reflection'] = 1.
        self.teapot['u_ambient'] = .11
        self.teapot['u_light_color'] = (1, 1, 1)
        self.teapot['u_light_intensity'] = 1.
        self.teapot['u_light_specular'] = (1, 50)
        self.teapot['u_light_position'] = (10, 10, 20)
        self.teapot['u_O'] = (0, 0, 10)

        self.theta, self.phi = 0, 0

        self.set_viewport()

        self.timer = app.Timer( 'auto', self.on_timer, start=True )

        gloo.set_state(depth_test=True)
        self.show()

    def on_draw ( self, event ):
        gloo.clear(color='black', depth=True)
        self.teapot.draw('triangles', self.FF)
        # draw normal lines
        #self.normals.draw('lines')

    def on_resize ( self, event ):
        self.set_viewport()

    def set_viewport ( self ):
        width, height = self.size
        self.teapot['u_proj'] = self.normals['u_proj'] = perspective( 45., width/float(height), 2., 20. )
        gloo.set_viewport( 0, 0, *self.physical_size )

    def on_timer ( self, event ):
        #self.theta += .5
        self.phi += .5
        self.teapot['u_model'] = self.normals['u_model'] = np.dot( rotate(self.theta, (0,0,1)),
                                                                   rotate(self.phi, (0,1,0) ))
        self.update()

if __name__ == '__main__':
    canvas = Canvas()
    app.run()
