import numpy as np

__all__ = [
    "FLOAT_EPSILON",
    "wrap",
    "sph2pix",
    "pix2sph",
]

# This error tolerance will maintain accuracy up to 10 meters in units of c
# or about 5 kilometers in astronomical units
FLOAT_EPSILON = 3.3356409519815204e-08


def wrap(angles, complement=False):
    """
    Wrap angles in radians to either the interval [0, 2*pi],
    or to the complement interval [-pi, pi].
    """
    angles = np.asarray(angles)
    if complement:
        return np.mod(angles + np.pi, 2 * np.pi) - np.pi
    return np.mod(angles + 2 * np.pi, 2 * np.pi)


def sph2pix(theta, phi, res):
    """
    Returns the coordinates of the pixel at the angular position (theta, phi).
    """
    x = np.rint((res[0] - 1) * wrap(phi) / (2 * np.pi))
    y = np.rint((res[1] - 1) * (1 - np.sin(theta)) / 2)
    return np.array([x, y]).astype(int)


def pix2sph(x, y, res):
    """
    Returns spherical angles in the domain [-pi/2, pi/2] x [-pi, pi].
    """
    phi = wrap(2 * np.pi * x / (res[0] - 1), complement=True)
    theta = np.arccos((2 * y / (res[1] - 1)) - 1) - np.pi / 2
    return np.array([theta, phi])
