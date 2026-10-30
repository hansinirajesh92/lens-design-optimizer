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

def sag(y, R):
    """how far a spherical surface sits from flat at height y (mm)

    R: radius (mm). R > 0 means the center is to the right, R < 0 to the left,
    np.inf means flat (sag is 0 everywhere). works on a number or a numpy array
    """

    if np.isinf(R):
        return 0 * y
    return R - np.sign(R) * np.sqrt(R ** 2 - y ** 2)


def normal_angle(y, R):
    """angle of the surface normal at height y, measured from the +z direction (radians)

    the normal points along the radius, through the center. flat surfaces give 0
    """

    if np.isinf(R):
        return 0 * y
    return - np.arcsin(y / R)


def refract_at(u, y, R, n1, n2):
    """new angle of a ray after snell's law at a spherical surface (radians, from +z)

    u: ray angle before, y: height where it hits (mm), R: surface radius (mm)
    n1: index before the surface, n2: index after
    works for any ray (flat or tilted) and any surface (curved or flat)
    """

    phi = normal_angle(y, R)
    theta2 = np.arcsin(n1 * np.sin(u - phi) / n2)
    return phi + theta2


def hit_surface(z0, y0, u, z_v, R):
    """where a ray meets a surface, found by guess and correct (z and y, in mm)

    (z0, y0): a point the ray passes through, u: its angle (radians)
    z_v: the surface's vertex position (mm), R: its radius (mm)
    starts by guessing the vertex, then moves to the surface 5 times
    """

    z = z_v
    for i in range(5):
        y = y0 + np.tan(u) * (z - z0)
        z = z_v + sag(y, R)
    y = y0 + np.tan(u) * (z - z0)
    return z, y


def trace(h, surfaces):
    """trace flat rays at heights h through a list of surfaces, in order

    surfaces: list of (z_v, R, n_after) for each surface the light meets
    returns where each ray leaves the last surface (z, y in mm) and its angle u (radians)
    """

    h = np.asarray(h, dtype = float)
    z = np.full_like(h, -20.0)
    y = h.copy()
    u = 0 * h
    n1 = 1.0
    for z_v, R, n2 in surfaces:
        z, y = hit_surface(z, y, u, z_v, R)
        u = refract_at(u, y, R, n1, n2)
        n1 = n2
    return z, y, u


def cross(h, surfaces):
    """where rays at heights h cross the optical axis after the lens (z, in mm)"""

    z, y, u = trace(h, surfaces)
    return z - y / np.tan(u)


def best_spot(surfaces, rays = np.linspace(-10, 10, 1001)):
    """find the sharpest sensor position for a lens by trying 501 positions from 35 to 60 mm

    rays: ray heights to trace (mm), 1001 rays from -10 to 10 unless given
    returns the best sensor position (mm) and its rms spot size (mm)
    """

    z_last, y, u = trace(rays, surfaces)
    sensors = np.linspace(35, 60, 501)
    spots = []
    for z in sensors:
        y_sensor = y + np.tan(u) * (z - z_last)
        spots.append(np.sqrt(np.mean(y_sensor ** 2)))
    spots = np.array(spots)
    best = np.argmin(spots)
    return sensors[best], spots[best]