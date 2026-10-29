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

def exit_flat(m1, n):
    """slope of a ray after it leaves glass through a flat surface

    m1: slope of the ray inside the glass, n: refractive index of the glass
    the flat surface's normal is horizontal, so the ray's own angle is the angle of incidence
    """

    u1 = np.arctan(m1)
    u2 = np.arcsin(n * np.sin(u1))
    return np.tan(u2)

def trace_lens(h, R, n, t):
    """trace a flat ray through a plano-convex lens (curved front, flat back)

    h: ray height (mm), R: front surface radius (mm), n: refractive index, t: thickness (mm)
    returns the ray's height at the back surface (mm) and its slope after leaving the lens
    works on a single height or a numpy array of heights
    """

    z1 = hit_z(h, R)
    m1 = refract_flat(h, R, 1.0, n)
    y2 = h + m1 * (t - z1)
    m2 = exit_flat(m1, n)
    return y2, m2

def cross_axis_back(y2, m2, t):
    """where a ray leaving the back of the lens crosses the optical axis (z, in mm)

    y2: height at the back surface (mm), m2: slope after the lens, t: thickness (mm)
    """

    return t - (y2 / m2)

def rms_spot(z, h, R, n, t):
    """rms spot size (mm) of rays through a plano-convex lens, measured at a sensor at z

    z: sensor position (mm), h: array of ray heights (mm)
    R: front radius (mm), n: refractive index, t: thickness (mm)
    """

    y2, m2 = trace_lens(h, R, n, t)
    y = y2 + m2 * (z - t)
    return np.sqrt(np.mean(y ** 2))