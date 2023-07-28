import numpy as np


def update_active_points(y, ykmin, ykmax, active_points, vertices, mk):
    """
    Updates the active points of each scanning line during the filling algorithm.

    param y: The height of the current scanning line (MxNx3).
    param ykmin: Array that holds maximum height of each triangle (1x3).
    param ykmin: Array that holds minimum height of each triangle (1x3).
    param active_points: Array that holds the coordinates and the edge of each current active point.
    param vertices: Holds the coordinates of the triangle's vertices (3x2).
    param mk: Array that holds the slope of each edge of the triangle
    return: The updated array of active points.
    """

    # update x of every old active point
    for point in active_points:
        k = point[2]
        if mk[k] != 0 and mk[k] != np.inf:
            dx = 1 / mk[k]
            point[0] += dx
        point[1] += 1

    for k in range(3):
        start_point = vertices[k]
        end_point = vertices[(k + 1) % 3]

        # delete old edges that are below y and their points
        if ykmax[k] == y:
            # remove points from that edge
            active_points = [point for point in active_points if point[2] != k]

        # add new edges and their points
        if ykmin[k] == (y + 1):
            if start_point[1] == (y + 1):
                if mk[k] != 0:
                    active_points.append([start_point[0], start_point[1], k])
            elif end_point[1] == (y + 1):
                if mk[k] != 0:
                    active_points.append([end_point[0], end_point[1], k])

    return active_points
