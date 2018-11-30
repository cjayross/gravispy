import numpy as np
from vispy import app, gloo
from vispy.geometry import create_sphere
from vispy.util.transforms import perspective
from vispy.io import read_png

VSHADER = """
uniform mat4 projection;
uniform sampler2D texture;

attribute vec3 position;
attribute vec2 texcoord;

varying vec2 v_texcoord;
void main()
{
    gl_Position = projection * vec4(position,1.0);
    v_texcoord = texcoord;
}
"""

FSHADER = """
uniform sampler2D texture;
varying vec2 v_texcoord;
void main()
{
    gl_FragColor = texture2D(texture, v_texcoord);
}
"""

class Canvas (app.Canvas):
    def __init__(self):
        # initialize the Canvas
        app.Canvas.__init__(self, size=(800,600), keys='interactive')
        # compile the programs for the blackhole and backdrop
        self.program = gloo.Program(VSHADER, FSHADER)

        mesh = create_sphere(2048, 1025, 1.)
        vertices = mesh.get_vertices('faces')
        self.indices = gloo.IndexBuffer(mesh.get_faces())
        self.program['position'] = vertices
        self.program.bind(gloo.VertexBuffer(vertices))

        projection = perspective(45.0, self.size[0] / float(self.size[1]),
                                 1.0, 10.0)
        self.program['projection'] = projection
        self.program['texcoord'] = [(0.,0.), (1.,0.),(0.,1.), (1.,0.), (0.,1.), (1.,1.)]
        self.program['texture'] = gloo.Texture2D(read_png('earth.png'))

        self.set_viewport()
        self.show()

    def on_resize(self, event):
        self.set_viewport()

    def set_viewport(self):
        width, height = self.size
        gloo.set_viewport(0, 0, *self.physical_size)

    def on_draw(self, event):
        gloo.clear(color='black', depth=True)
        self.program.draw('triangles', self.indices)

if __name__ == '__main__':
    canvas = Canvas()
    app.run()
