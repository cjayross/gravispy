from distutils.core import setup

setup(
        name='gravispy',
        description='Gravitational lensing using exact lensing equations for static, spherical spacetimes.',
        url='https://github.com/cjayross/gravispy',
        version='0.1-dev',
        packages=['gravispy',],
        install_requires=['numpy', 'sympy', 'scipy', 'PIL', 'riccipy'],
        )
