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

def recolour(surfaces, n):
    """the same lens with a different refractive index for its glass, to trace another colour

    surfaces: list of (z_v, R, n_after), n: the glass's index for the new colour
    every n_after that isn't air (1.0) becomes n; the shape stays exactly the same
    """

    new = []
    for z_v, R, n_after in surfaces:
        if n_after == 1.0:
            new.append((z_v, R, 1.0))
        else:
            new.append((z_v, R, n))
    return new
