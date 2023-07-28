import numpy as np

from interpolate_vectors import interpolate_vectors
from lighting import light
from update_active_points import update_active_points


def shade_phong(vertsp, vertsn, vertsc, bcoords, cam_pos, mat, lights, light_amb, light_type, X):
    """
    Renders the passed triangle on the canvas using the "phong" method.

    param vertsp: 2 × 3 array containing the coordinates of the triangle's vertices after they are projected onto the
    camera curtain.
    param vertsn: 3 × 3 matrix containing the normal vectors of the vertices of the triangle.
    param vertsc: 3 × 3 array containing the color components for each point of the triangle.
    param bcoords: 3 × 1 vector containing the center of gravity of the triangle before projection.
    param cam_pos: 3 × 1 column vector of observer (camera) coordinates.
    param mat : Object of type PhongMaterial holding material properties.
    param lights: List of PointLight objects representing the light sources.
    param light_amb: 3 × 1 vector of diffuse intensity components representing environmental radiation.
    param light_type: Sets the lighting type to "ambient", "diffuse", "specular" or "all".
    param X: Image matrix of dimension M × N × 3 with any preexisting triangles.
    return: The updated canvas with the triangle shaded.
    """

    vertsp = vertsp.T
    vertsn = vertsn.T
    vertsc = vertsc.T

    ykmin = [0 for _ in range(3)]
    ykmax = [0 for _ in range(3)]

    # we find min, max for each edge k = [0, 2]
    for k in range(3):
        ykmin[k] = min(vertsp[k][1], vertsp[(k + 1) % 3][1])
        ykmax[k] = max(vertsp[k][1], vertsp[(k + 1) % 3][1])

    ymin = min(ykmin)
    ymax = max(ykmax)

    active_points = []
    mk = [0.0 for _ in range(3)]

    # we find the list of active points
    for k in range(3):
        start_point = vertsp[k]
        end_point = vertsp[(k + 1) % 3]

        # grad
        if (end_point[0] - start_point[0]) == 0.0:
            mk[k] = np.inf
        else:
            mk[k] = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])

        if start_point[1] == ymin:
            if mk[k] != 0:
                active_points.append([start_point[0], start_point[1], k])
        if end_point[1] == ymin:
            if mk[k] != 0:
                active_points.append([end_point[0], end_point[1], k])

    # loop for each y horizontal line between ymin and ymax
    for y in range(ymin, ymax):
        # sort active points by x
        active_points = sorted(active_points, key=lambda l: l[0])

        # calculate A, B and their colors for line y
        A, B, A_color, B_color, A_normal, B_normal = calculate_colors_normals(active_points, vertsp, vertsc, vertsn)

        for x in range(round(active_points[0][0]), round(active_points[-1][0])):
            color = interpolate_vectors([A[0], A[1]], [B[0], B[1]], A_color, B_color, x, 1)
            normal = interpolate_vectors([A[0], A[1]], [B[0], B[1]], A_normal, B_normal, x, 1)

            color = np.array([color]).T
            normal = np.array([normal]).T

            # Calculates the color of the point using the full light model
            color = light(bcoords, normal, color, cam_pos, mat, lights, light_amb, light_type).T[0]

            color = np.clip(color, 0, 1)
            X[y][x] = color

        # update active points
        active_points = update_active_points(y, ykmin, ykmax, active_points, vertsp, mk)

    return X


def calculate_colors_normals(active_points, vertices, vcolors, normals):
    """
    Calculates the coordinates, the color and the normal vectors of the two points that the scanning line crosses the
    triangle, using the vector interpolation function.

    param active_points: Array that holds the coordinates and the edge of each current active point.
    param vertices: Holds the coordinates of the triangle's vertices (3x2).
    param vcolors: Holds the RGB colors of each vertex (Lx3).
    param normals: Holds the normal vector of each vertex (Lx3).
    return: The coordinates and the color of each point.
    """

    # get the first and the last point of active_points array
    A = active_points[0]
    B = active_points[-1]

    A_start = A[2]
    A_end = (A_start + 1) % 3
    A_color = interpolate_vectors(vertices[A_start], vertices[A_end], vcolors[A_start], vcolors[A_end], A[1], 2)
    A_normal = interpolate_vectors(vertices[A_start], vertices[A_end], normals[A_start], normals[A_end], A[1], 2)

    B_start = B[2]
    B_end = (B_start + 1) % 3
    B_color = interpolate_vectors(vertices[B_start], vertices[B_end], vcolors[B_start], vcolors[B_end], B[1], 2)
    B_normal = interpolate_vectors(vertices[B_start], vertices[B_end], normals[B_start], normals[B_end], B[1], 2)

    return A, B, A_color, B_color, A_normal, B_normal
