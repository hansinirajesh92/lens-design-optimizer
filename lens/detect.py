import numpy as np
from lens.raytrace import trace, cross, best_spot

def spherical(surfaces, a = 5.0):
    """longitudinal spherical aberration (mm): how much earlier an edge ray crosses the axis than a near-center ray

    surfaces: list of (z_v, R, n_after), a: height of the edge ray (mm), the edge of the opening
    negative means the edge focuses closer to the lens (the usual case for a simple lens)
    """

    edge = cross(a, surfaces)
    center = cross(0.01, surfaces)
    return edge - center

def field_curvature(surfaces, deg = 10, rays = np.linspace(-5, 5, 1001), z_stop = 0.0):
    """field curvature (mm): how far the best focus moves between the center of the image and an edge angle

    surfaces: list of (z_v, R, n_after), deg: edge field angle (degrees)
    rays: ray heights at the stop (mm), f/5 unless given, z_stop: aperture stop position (mm)
    negative means the edge focuses closer to the lens (the image bowl curves toward the lens)
    """

    z_center, spot_center = best_spot(surfaces, 0.0, rays, z_stop)
    z_edge, spot_edge = best_spot(surfaces, np.radians(deg), rays, z_stop)
    return z_edge - z_center

def coma(surfaces, deg = 10, a = 5.0, z_stop = 0.0):
    """tangential coma (mm): how far the dot's two edge rays land off-center from its middle ray

    surfaces: list of (z_v, R, n_after), deg: field angle (degrees)
    a: height of the edge rays at the stop (mm), z_stop: aperture stop position (mm)
    measured at that angle's own best focus. 0 means a symmetric dot (no coma),
    the sign says which way the comet tail points
    """

    angle = np.radians(deg)
    z_sensor, spot = best_spot(surfaces, angle, np.linspace(-a, a, 1001), z_stop)
    z_last, y, u = trace(np.array([-a, 0.0, a]), surfaces, angle, z_stop)
    y_sensor = y + np.tan(u) * (z_sensor - z_last)
    return (y_sensor[0] + y_sensor[2]) / 2 - y_sensor[1]
