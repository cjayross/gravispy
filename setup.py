from distutils.core import setup

setup(
        name='gravispy',
        description='Gravitational simulations using vispy',
        long_description=open('README.md').read(),
        url='https://github.com/cjayross/gravispy',
        version='0.1-dev',
        packages=['gravispy',],
        install_requires=['numpy', 'sympy', 'scipy'],
        extras_require = {'planned implementations' : 'pyopencl'}
        )
