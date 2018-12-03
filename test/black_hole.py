import numpy as np
from pathlib import Path
from vispy import app, gloo, io
from vispy.util.transforms import perspective, rotate, translate

VSHADER = """
#version 120

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

attribute vec3 a_position;
attribute vec3 a_texcoord;

varying vec2 v_texcoord;
void main()
{
    gl_Position = u_proj*u_view*u_model*vec4(a_position, 1.);
    v_texcoord = a_texcoord.xy;
}
"""

FSHADER = """
uniform sampler2D u_sampler;

varying vec2 v_texcoord;
void main()
{
    gl_FragColor = texture2D(u_sampler, v_texcoord);
}
"""

def create_sphere():
    vtype = [('a_position', np.float32, 3),
             ('a_texcoord', np.float32, 3)]
    V, F, N, T = io.read_mesh(Path('./obj/sphere.obj'))
    f_buf = gloo.IndexBuffer(F)
    v_buf = np.zeros(len(V), dtype=vtype)
    v_buf['a_position'] = V
    v_buf['a_texcoord'] = T
    v_buf = gloo.VertexBuffer(v_buf)
    return v_buf, f_buf

def lookat(th, ph):
    return np.dot(rotate(th,(1,0,0)),rotate(ph,(0,1,0)))

class Canvas (app.Canvas):
    def __init__(self):
        app.Canvas.__init__(
                self,
                keys='interactive',
                size=(800,600),
                config=dict(samples=4)
                )

        self.V, self.F = create_sphere()
        self.program = gloo.Program(VSHADER, FSHADER)
        self.program.bind(self.V)
        self.program['u_sampler'] = gloo.Texture2D(
                io.read_png('earth.png'),
                interpolation='linear')

        self.th, self.ph = 0., 0.
        self.delta = 5.
        self.fov = 80.

        self.view = lookat(self.th, self.ph)
        self.model = rotate(180., (1,0,0))

        gloo.set_state(depth_test=True)
        self.set_viewport()
        self.show()

    def on_resize(self, event):
        self.set_viewport()

    def on_key_press(self, event):
        if event.text == 'j':
            self.th += self.delta
        elif event.text == 'k':
            self.th -= self.delta
        elif event.text == 'h':
            self.ph -= self.delta
        elif event.text == 'l':
            self.ph += self.delta
        elif event.text == 'n':
            width, height = self.size
            self.fov += self.delta/2
            self.fov = np.clip(self.fov, 10., 90.)
            self.proj = perspective(self.fov, np.divide(width, height),
                                    1., 100.)
        elif event.text == 'p':
            width, height = self.size
            self.fov -= self.delta/2
            self.fov = np.clip(self.fov, 10., 90.)
            self.proj = perspective(self.fov, np.divide(width, height),
                                    1., 100.)
        self.view = lookat(self.th, self.ph)
        self.update()

    def set_viewport(self):
        width, height = self.size
        self.proj = perspective(self.fov, np.divide(width, height),
                                1., 100.)
        gloo.set_viewport(0, 0, *self.physical_size)

    def on_draw(self, event):
        gloo.clear(color='black', depth=True)
        self.program.draw('triangles', self.F)

    @property
    def model(self):
        return self._model

    @property
    def view(self):
        return self._view

    @property
    def proj(self):
        return self._proj

    @model.setter
    def model(self, mat):
        self._model = self.program['u_model'] = mat

    @view.setter
    def view(self, mat):
        self._view = self.program['u_view'] = mat

    @proj.setter
    def proj(self, mat):
        self._proj = self.program['u_proj'] = mat

if __name__ == '__main__':
    canvas = Canvas()
    app.run()
