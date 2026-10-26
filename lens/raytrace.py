import numpy as np

def hit_z(h, R):
    return R - np.sqrt(R ** 2 - h ** 2)

def refract_flat(h, R, n1, n2):
    alpha = np.arcsin(h / R)
    theta2 = np.arcsin((n1 * np.sin(alpha)) / n2)
    u = -(alpha - theta2)
    return np.tan(u)

def crossing(h, R, n1, n2):
    m = refract_flat(h, R, n1, n2)
    return hit_z(h, R) - h / m