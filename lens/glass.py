catalog = [
    {"name": "N-FK51A", "nd": 1.48656, "V": 84.47},
    {"name": "N-BK7",   "nd": 1.51680, "V": 64.17},
    {"name": "N-K5",    "nd": 1.52249, "V": 59.48},
    {"name": "N-SK16",  "nd": 1.62041, "V": 60.32},
    {"name": "N-F2",    "nd": 1.62005, "V": 36.43},
    {"name": "N-LAK9",  "nd": 1.69100, "V": 54.71},
    {"name": "N-SF11",  "nd": 1.78472, "V": 25.68},
]


def colours(glass):
    """a glass's refractive index for blue and red light, from its catalog nd and abbe number V

    glass: a dictionary with "nd" (index for yellow-green light) and "V" (abbe number)
    the blue-red spread is (nd - 1) / V, split evenly either side of nd
    returns n_blue, n_red
    """

    spread = (glass["nd"] - 1) / glass["V"]
    n_blue = glass["nd"] + spread / 2
    n_red = glass["nd"] - spread / 2
    return n_blue, n_red
