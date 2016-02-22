import math


def hedges_g(x1, s1, n1, x2, s2, n2):
    pooled_sd = math.sqrt(float((((n1 - 1) * s1 ** 2) +
                                ((n2 - 1) * s2 ** 2)) / (n1 + n2 - 2)))
    cohens_d = float(x1 - x2) / pooled_sd
    correction_factor = 1 - (3.0 / (4 * (n1 + n2) - 9))
    return cohens_d * correction_factor


print hedges_g(87, 10, 100, 60, 25, 8)



