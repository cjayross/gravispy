from sys import argv
from pathlib import Path
from PIL import Image
from gravispy import *

r_obs = float(argv[1]) if len(argv) > 1 else 30.0
r_src = 1e4 * r_obs
schwarzschild = metrics.Schwarzschild(mass=1.0)

img = Image.open(Path("png/blue_marble.png"))

mapping = generate_lens_map(
    lensing.spherical_lens, img.size, args=(schwarzschild, r_obs, r_src)
)
new = apply_lensing(img, mapping)
new.save(Path("png/example_lensing.png"))
