import numpy as np
import itertools as it
from PIL import Image
from gravispy.util import pix2sph, sph2pix, wrap

__all__ = [
    'generate_lens_map',
    'apply_lensing',
]

def generate_lens_map(lens, res, args=(), kwargs={}, prec=3):
    coords = list(it.product(*map(np.arange, res)))
    x, y = np.asarray(coords).astype(int).T
    theta, phi = pix2sph(x, y, res)

    arccos2 = lambda a: np.sign(wrap(a, complement=True))*np.arccos(a)
    arcsin2 = lambda a, k: np.pi*k + (-1)**k*np.arcsin(a)

    # consider alphas to be equal if they are the same up
    # to 3 decimals (by default). this reduces the amount of
    # calls to lens from possibly millions to only about 3000
    alpha = np.round(arccos2(np.cos(theta)*np.cos(phi)), prec)
    # compress alpha
    alphaz = np.unique(alpha)
    betaz = np.fromiter(lens(alphaz, *args, **kwargs), np.float64)
    # expand betaz
    beta_map = dict(zip(alphaz, betaz))
    beta = np.fromiter(map(beta_map.__getitem__, alpha), np.float64)

    # we will intentionally fail invalid calculations,
    # they will be filtered out afterward.
    # as such, we don't need to be warned that they occurred.
    errstate = np.seterr(all='ignore')

    sigma = np.sin(beta) / np.sin(alpha)
    mu, nu = map(lambda a: sigma*np.sin(a), [theta, phi])
    # this choice of k's needs to be scrutinized
    k1, k2 = map(lambda a: np.abs(wrap(a, complement=True)) > np.pi/2, [theta, phi])
    psi, gamma = map(arcsin2, [mu, nu], [k1,k2])

    # cut out invalid results
    idxs = np.logical_not(np.isnan(psi) | np.isnan(gamma))
    keys = zip(x[idxs], y[idxs])
    values = zip(*sph2pix(psi[idxs], gamma[idxs], res))

    np.seterr(**errstate)

    return dict(zip(keys, values))

def apply_lensing(img, lens_map, res=None):
    if not res:
        res = img.size
    pix = img.load()
    new = Image.new(img.mode, res)
    new_pix = new.load()
    for pix_coord in it.product(*map(range, res)):
        # we currently aren't implementing hidden images
        try:
            map_coord = tuple(map(int, lens_map[pix_coord]))
            new_pix[pix_coord] = pix[map_coord]
        except KeyError:
            # the pixel defaults to black
            continue
    return new
