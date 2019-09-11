import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from pathlib import Path
from gravispy import *

r_obs = float(argv[1]) if len(argv) > 1 else 10.0
r_src = 1e+4 * r_obs
schwarzschild = metrics.Schwarzschild(mass=1.0)

thetas = np.linspace(-np.pi, np.pi, 1000)
phis = np.fromiter(lensing.spherical_lens(thetas, schwarzschild, r_obs, r_src), dtype=np.float32)

plt.xlabel(r'Observation Angle, $\Theta$')
plt.ylabel(r'Source Angle, $\Phi$')
plt.title(r'Schwarzschild Lensing ($R_O = {}M$)'.format(r_obs))
plt.xlim(-np.pi, np.pi)
plt.plot(thetas, phis)
plt.savefig(Path('png/plot_result.png'))
