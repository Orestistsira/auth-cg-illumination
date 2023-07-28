import numpy as np


def interpolate_vectors(p1, p2, V1, V2, xy, dim):
    """
    Calculates the value of a vector at the position p by linear interpolation between the vectors V1 and V2
    corresponding to the positions p1 and p2 with the assumption that the point p belongs to the line segment p1p2.

    param p1: Tuple with the two-dimensional coordinates of the first point.
    param p2: Tuple with the two-dimensional coordinates of the second point.
    param V1: Vector value at position p1.
    param V2: Vector value at position p2.
    param xy: Abscissa (x) or ordinate (y) of the point p depending on whether dim = 1 or dim = 2 respectively.
    param dim: Dimension to interpolate (1 for x, 2 for y).
    return: Interpolated vector value at position p.
    """

    if dim == 1:
        # Interpolate along the x-axis
        if (p2[0] - p1[0]) == 0:
            return V1
        t = (xy - p1[0]) / (p2[0] - p1[0])
        V = V1 + t * np.subtract(V2, V1)
    elif dim == 2:
        # Interpolate along the y-axis
        if (p2[1] - p1[1]) == 0:
            return V1
        t = (xy - p1[1]) / (p2[1] - p1[1])
        V = V1 + t * np.subtract(V2, V1)
    else:
        raise ValueError("Invalid value for dim. Expected 1 or 2.")
    return V
