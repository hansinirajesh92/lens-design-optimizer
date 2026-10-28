import numpy as np

def hit_z(h, R):
    """where a flat ray at height h meets a spherical surface

    the surface's vertex is at z = 0 and its center is at z = R
    works on a single height or a numpy array of heights
    """

    return R - np.sqrt(R ** 2 - h ** 2)

def refract_flat(h, R, n1, n2):
    """slope of a flat ray after it refracts at a spherical surface

    h: ray height, R: surface radius, n1/n2: index before/after
    uses snell's law; angles are in radians internally
    """

    alpha = np.arcsin(h / R)
    theta2 = np.arcsin((n1 * np.sin(alpha)) / n2)
    u = -(alpha - theta2)
    return np.tan(u)

def crossing(h, R, n1, n2):
    """where the refracted ray crosses the optical axis (z, in mm)"""

    m = refract_flat(h, R, n1, n2)
    return hit_z(h, R) - h / m