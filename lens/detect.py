import numpy as np
from lens.raytrace import trace, cross, best_spot
from lens.design import recolour
from lens.glass import colours

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

def diagnose(surfaces, a = 5.0, deg = 10, z_stop = 0.0, f = 50.0, glass = None):
    """run the detectors on a lens, rank them by how much blur each causes, and explain the biggest

    surfaces: list of (z_v, R, n_after), a: edge of the opening (mm, 5 means f/5)
    deg: edge field angle (degrees), z_stop: aperture stop position (mm), f: focal length (mm)
    glass: catalog dictionary (optional, adds the chromatic detector)
    along-axis errors become blur with: blur ≈ error × (a / f), a rough estimate for ranking
    prints a short report and returns the name of the main problem
    """

    sa = spherical(surfaces, a)
    fc = field_curvature(surfaces, deg, np.linspace(-a, a, 1001), z_stop)
    cm = coma(surfaces, deg, a, z_stop)

    slope = a / f
    names = ["spherical", "field curvature", "coma"]
    blurs = [abs(sa) * slope, abs(fc) * slope, abs(cm)]
    print(f"spherical:        {sa:+.3f} mm along the axis  → about {blurs[0]:.3f} mm of blur")
    print(f"field curvature:  {fc:+.3f} mm along the axis  → about {blurs[1]:.3f} mm of blur")
    print(f"coma:             {cm:+.4f} mm on the sensor   → about {blurs[2]:.3f} mm of blur")

    if glass is not None:
        n_blue, n_red = colours(glass)
        ch = chromatic(surfaces, n_blue, n_red)
        names.append("chromatic")
        blurs.append(abs(ch) * slope)
        print(f"chromatic:        {ch:+.3f} mm along the axis  → about {blurs[3]:.3f} mm of blur")

    blurs = np.array(blurs)
    main = names[np.argmax(blurs)]
    print("main problem:", main)
    if main == "spherical":
        print("rays through the edge of the lens focus closer than rays through the middle, even straight ahead.")
        print("try: a shape closer to best-form, a smaller opening, or a glass with a higher index.")
    elif main == "field curvature":
        print("the edges of the image focus closer to the lens than the center, so a flat sensor can't catch both.")
        print("try: moving the aperture stop, bending the lens, or a compromise sensor position.")
    elif main == "coma":
        print("off-axis dots flare out to one side like a comet.")
        print("try: moving the aperture stop or bending the lens.")
    else:
        print("blue light focuses closer to the lens than red, so white light smears into coloured fringes.")
        print("try: a glass with a bigger abbe number V (it spreads colours less).")
    return main

def chromatic(surfaces, n_blue, n_red):
    """longitudinal chromatic aberration (mm): how far blue light focuses from red light

    surfaces: list of (z_v, R, n_after), n_blue / n_red: the glass's index for blue and red light
    uses near-axis rays (h = 0.01 mm) so spherical aberration doesn't mix in
    negative means blue focuses closer to the lens (the usual case)
    """

    blue = recolour(surfaces, n_blue)
    red = recolour(surfaces, n_red)
    return cross(0.01, blue) - cross(0.01, red)
