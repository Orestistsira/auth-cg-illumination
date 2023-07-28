import numpy as np

from shade_gouraud import shade_gouraud
from shade_phong import shade_phong


def render(verts2d, normals, faces, vcolors, bcoords, cam_pos, mat, lights, light_amb, depth, bg_color, shade_t, light_type):
    """
    Sorts triangles by their depth and passes them and draws them on the canvas from the furthest to the
    closest to the camera.

    param verts2d: Holds the coordinates of each vertex (Lx2).
    param normals: An array that contains the normal vector of every triangle's vertices.
    param faces: Holds the indexes from verts2d array of the vertices for each triangle (Kx3).
    param vcolors: Holds the RGB colors of each vertex (Lx3).
    param bcoords: An array that contains barycentric coordinates of every triangle's vertices
    param cam_pos: A 3 x 1 vector with the coordinates of the camera's center.
    param mat: An object of type PhongMaterial that holds the material properties for the object.
    param lights: A list of PointLight objects representing the light sources in the scene.
    param light_amb: A 3 x 1 vector of diffuse intensity components representing the environmental radiation.
    param bg_color: A 1 x 3 vector that contains the background color.
    param depth: Holds the distance from the camera for each vertex (Lx1).
    param shade_t: Sets the shading method to "phong" or "gouraud".
    param light_type: Sets the lighting type to "ambient", "diffuse", "specular" or "all".
    return: The image with the triangles rendered.
    """

    if shade_t not in ["phong", "gouraud"]:
        print('ERROR: Invalid shading method "' + str(shade_t) + '", choose from ["phong", "gouraud"]')
        exit(1)

    cam_pos = np.array([cam_pos]).T

    # init image with bg_color
    M = 512
    N = 512
    img = [[bg_color for _ in range(N)] for _ in range(M)]

    # number of triangles
    K = len(faces)

    # sort triangle faces from farthest from the camera to closest
    triangle_depth = [0.0 for _ in range(K)]

    for triangle in range(K):
        triangle_depth[triangle] = sum(depth[faces[triangle]]) / 3

    # sort in descending order
    sorted_indexes = np.argsort(triangle_depth)[::-1]

    # faces sorted
    faces = faces[sorted_indexes]

    for triangle in range(K):
        tr_indices = faces[triangle]
        tr_vertices = verts2d[tr_indices].T
        tr_vcolors = vcolors[tr_indices].T
        tr_normals = normals[:, tr_indices]
        tr_bcoords = np.array([bcoords[:, triangle]]).T

        img = shade_triangle(img, tr_vertices, tr_normals, tr_vcolors, tr_bcoords,
                             cam_pos, mat, lights, light_amb, shade_t, light_type)

    return img


def shade_triangle(canvas, vertices, normals, vcolors, bcoords, cam_pos, mat, lights, light_amb, shade_t, light_type):
    """
    Calls the corresponding shading function based on shade_t parameter for the input triangle.

    param canvas: Holds the image data (MxNx3).
    param vertices: Holds the coordinates of the triangle's vertices (3x2).
    param normals: An array that contains the normal vector of every triangle's vertices.
    param vcolors: Holds the RGB colors of each vertex of the triangle (3x3).
    return: The updated canvas with the triangle shaded.
    param bcoords: An array that contains barycentric coordinates of every triangle's vertices
    param cam_pos: A 3 x 1 vector with the coordinates of the camera's center.
    param mat: An object of type PhongMaterial that holds the material properties for the object.
    param lights: A list of PointLight objects representing the light sources in the scene.
    param light_amb: A 3 x 1 vector of diffuse intensity components representing the environmental radiation.
    param shade_t: Sets the shading method to "phong" or "gouraud".
    param light_type: Sets the lighting type to "ambient", "diffuse", "specular" or "all".
    return: The image with the input triangle rendered.
    """

    if shade_t == "phong":
        img = shade_phong(vertices, normals, vcolors, bcoords, cam_pos, mat, lights, light_amb, light_type, canvas)
    else:
        img = shade_gouraud(vertices, normals, vcolors, bcoords, cam_pos, mat, lights, light_amb, light_type, canvas)

    return img
