import numpy as np


def R2_for(R1, t, f, n = 1.5):
    """back radius (mm) that gives a lens focal length f, from the thick-lens lensmaker's equation

    R1: front radius (mm), t: thickness (mm), f: target focal length (mm), n: refractive index
    returns np.inf when the back comes out flat
    """

    c1 = 1 / R1
    c2 = (1 / (f * (n - 1)) - c1) / ((n - 1) * t * c1 / n - 1)
    if abs(c2) < 1e-9:
        return np.inf
    return 1 / c2