# gravispy
> Gravitational lensing simulations written in Python.

## Overview

`gravispy` is a small python module providing methods for simulating the
distortion effects of gravitational lensing using physically accurate lensing
equations.

The functions of interest are `generate_lens_map` and `apply_lensing` (see
below for example usage). The submodules `metrics` and `lensing` are
also used to encapsulate the available metrics and lensing algorithms.  As of
now, only Schwarzschild spacetimes are supported.

### Example

```python
from pathlib import Path
from PIL import Image
from gravispy import metrics, lensing, generate_lens_map, apply_lensing

schwarzschild = metrics.Schwarzschild(mass=1.0)
img = Image.open(Path('blue_marble.png'))
# The lens map is generated from a separate function call in case the results
# are intended to be used more than once. `generate_lens_map` is a very time
# consuming function.
mapping = generate_lens_map(
    lensing.spherical_lens, img.size, args=(schwarzschild, 30.0, 30e4)
)

new_img = apply_lensing(img, mapping)
new_img.save(Path('example_lensing.png'))
```

#### Input
![input image](https://github.com/cjayross/gravispy/blob/master/examples/png/blue_marble.png?raw=true)
#### Output
![output image](https://github.com/cjayross/gravispy/blob/master/examples/png/example_lensing.png?raw=true)

## Requirements

+ Sympy (version >= 1.4)

+ Numpy (version >= 1.15)

+ RicciPy (version >= 0.2)

+ SciPy (version >= 1.4)

## References

+ [On the exact gravitational lens equation in spherically symmetric and static spacetimes](https://arxiv.org/pdf/gr-qc/0307072.pdf)
+ [Spacetime perspective of Schwarzschild lensing](https://arxiv.org/pdf/gr-qc/0001037.pdf)
