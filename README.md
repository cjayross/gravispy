# gravispy
Gravitational simulations written in Python.

### Requirements

+ numpy
+ vispy
+ sympy
+ scipy

### References

The following link is a very helpful pointer toward the sort of tools we might consider using:

<https://arxiv.org/pdf/1703.09738.pdf>

Although the topics it mentions are a little advanced, it is noted that [gravipy](https://pypi.org/project/GraviPy/) is a very fast module for doing calculations necessary to be able to calculated the trajectories of null geodesics; specifically, note section 4.3 (Visualizing geodesics).
I still think we are better off for now simply hardcoding such data, and even more so only focusing on the Schwarzschild metric.

This link discusses the exact lensing equations that we are interested in:

<https://arxiv.org/pdf/gr-qc/0307072.pdf>

The following is more abstract, but still has a couple of tips to consider:

<https://arxiv.org/pdf/gr-qc/0001037.pdf>

### Installation

To install in developer mode, go to the base directory that you've cloned and execute the pip command:

```bash
pip install --user -e .
```

Now, you can access gravispy as you would any other installed module.
