from distutils.core import setup

setup(
    name="gravispy",
    description="Gravitational lensing using exact lensing equations for static, spherical spacetimes.",
    url="https://github.com/cjayross/gravispy",
    version="0.1a",
    packages=["gravispy",],
    install_requires=["numpy", "sympy", "scipy", "riccipy"],
)
