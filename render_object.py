import numpy as np

from render import render


def ChangeCoordinateSystem(cp, R, c0):
    """
    Transforms a point or an array of points cp in R^3 (in non-homogeneous form) to a new coordinate system dp using a
    rotation matrix R and a translation vector c0.

    param cp: Point or array of points in R^3 (in non-homogeneous form) with respect to the original coordinate system.
    param R: Rotation matrix from the original coordinate system to the new coordinate system.
    param c0: Translation vector representing the origin of the new coordinate system with respect to the original.
    return dp: Transformed point or array of points in R^3 (in non-homogeneous form) with respect to the new coordinate
    system.
    """

    # Compute the new point(s) in the new coordinate system by applying the rotation and translation transformations
    dp = R @ (cp - c0)

    return dp


def PinHole(f, cv, cx, cy, cz, p3d):
    """
    Computes the perspective view of a point or an array of points p3d in R^3 (in nonhomogeneous form) using a pinhole camera model.

    param f: Distance of the curtain from the center (measured in the units used by the camera coordinate system).
    param cv: Coordinates of the center of the perspective camera with respect to the WCS.
    param cx: Unit vector x of the perspective camera with respect to the WCS.
    param cy: Unit vector y of the perspective camera with respect to the WCS.
    param cz: Unit vector z of the perspective camera with respect to the WCS.
    param p3d: Point or array of points in R^3 (in nonhomogeneous form) with respect to the WCS.
    return p2d: Perspective view of the point or array of points p3d in 2D.
    return depth: Depth of each point in p3d in the perspective camera view.
    """

    # Define the camera coordinate system
    R = np.array([cx, cy, cz])
    c0 = cv.reshape(3, 1)

    # Transform the points from WCS to the camera coordinate system
    p_cam = ChangeCoordinateSystem(p3d, R, c0)

    # Compute depth of each point
    depth = p_cam[2, :]

    # Compute perspective projection
    # p2d = (f / depth) * p_cam[:2, :]
    p2d = f * (p_cam[:2, :] / depth)

    return p2d, depth


def CameraLookingAt(f, cv, cK, cup, p3d):
    """
    Computes the perspective views and depth of the 3D points p3d using a pinhole camera model,
    where the camera is looking at a target point K and the up vector is given.

    param f: Distance of the curtain from the center (measured in the units used by the camera coordinate system).
    param cv: Coordinates of the center of the camera with respect to the WCS.
    param cK: Coordinates of the target point K with respect to the WCS.
    param cup: Unit vector representing the up direction of the camera.
    param p3d: Point or array of points in R^3 (in nonhomogeneous form) with respect to the WCS.
    return p2d: Perspective view of the point or array of points p3d in 2D.
    return depth: Depth of each point in p3d in the perspective camera view.
    """

    # Compute the camera coordinate system
    zc = (cK - cv) / np.linalg.norm(cK - cv)
    t = cup - (cup @ zc) * zc
    yc = t / np.linalg.norm(t)
    xc = np.cross(yc, zc)

    # Call PinHole function to project 3D points
    p2d, depth = PinHole(f, cv, xc, yc, zc, p3d)

    # print('p2d:')
    # print(p2d)
    # print('depth:')
    # print(depth)

    return p2d, depth


def rasterize(p2d, Rows, Columns, H, W):
    """
    Converts the 2D camera coordinate points to their corresponding pixel positions in the image.

    param p2d: 2D camera coordinate points.
    param Rows: Number of rows in the image.
    param Columns: Number of columns in the image.
    param H: Height of the camera's curtain in inches.
    param W: Width of the camera's curtain in inches.
    return n2d: Pixel positions of the points in the image.
    """

    # Scale the camera coordinate points to match pixel positions
    scaled_x = ((p2d[0, :] + W / 2) / W) * Columns
    # scaled_y = ((H / 2 - p2d[1, :]) / H) * Rows
    scaled_y = ((p2d[1, :] + H / 2) / H) * Rows

    # Round the scaled coordinates to the nearest integers
    n2d = np.round(np.vstack((scaled_x, scaled_y))).astype(int)

    return n2d


def calculate_bcoords(verts, faces):
    """
    Calculates the barycentric coordinates of every triangle's vertices

    param verts: 3xNv array of the coordinates of the object's vertices.
    param faces: 3xNt  array describing the triangles. The k-th column of faces contains the ordinal numbers of the
    return bcoords: 3xNv the barycentric coordinates array
    """

    bcoords = np.mean(verts[:, faces], axis=0)

    return bcoords


def calculate_normals(verts, faces):
    """
    Calculates the normal vector of every triangle's vertices

    param verts: 3xNv array of the coordinates of the object's vertices.
    param faces: 3xNt  array describing the triangles. The k-th column of faces contains the ordinal numbers of the
    vertices of the k-th triangle of the object.
    return normals: 3xNv The normal vectors list of every vertex.
    """

    num_of_vertices = verts.shape[1]
    num_of_triangles = faces.shape[1]
    normals = np.zeros((3, num_of_vertices))

    for i in range(num_of_triangles):
        face = faces[:, i]

        edge_1 = verts[:, face[1]] - verts[:, face[0]]
        edge_2 = verts[:, face[2]] - verts[:, face[0]]

        face_normal = np.cross(edge_1, edge_2)
        normals[:, faces[:, i]] += np.array([face_normal]).T

    # Normalize the vertex normals
    normals /= np.linalg.norm(normals, axis=0)

    return normals


def render_object(shader, focal, eye, lookat, up, bg_color, M, N, H, W, verts, vert_colors, faces, mat, lights, light_amb, light_type):
    """
    Renders the object on the canvas

    param shader: A binary control variable used to select the function for shading the triangles. Possible values are
    "gouraud" for Gouraud shading and "phong" for Phong shading.
    param focal: The distance of the curtain from the center of the camera, measured in the units of measurement used by
    the camera's coordinate system.
    param eye: A 3 × 1 vector with the coordinates of the camera's center.
    param lookat: A 3 × 1 vector with the coordinates of the camera's target point. It defines the direction in which
    the camera is pointing.
    param up: A 3 × 1 unit vector representing the camera's up direction. It helps define the orientation of the camera.
    param bg_color: A 3 × 1 vector of color components representing the background color. Each component corresponds to
    the red, green, and blue channels, respectively.
    param M: The height (number of rows) of the generated image in pixels.
    param N: The width (number of columns) of the generated image in pixels.
    param H: The physical height of the camera curtain, measured in units of length identical to those used in the
    camera coordinate system.
    param W: The physical width of the camera curtain, measured in units of length identical to those used in the camera
    coordinate system.
    param verts: A 3 × Nv array of the coordinates of the object's vertices. Each column represents a vertex, and the
    rows contain the x, y, and z coordinates of the vertex, respectively.
    param vert_colors: A 3 × Nv array of the color components of each vertex of the object.
    param faces: The faces of the object, which define the triangles.
    param mat: An object of type PhongMaterial that holds the material properties for the object.
    param lights: A list of PointLight objects representing the light sources in the scene.
    param light_amb: A 3 × 1 vector of diffuse intensity components representing the environmental radiation.
    param light_type: Sets the lighting type to "ambient", "diffuse", "specular" or "all".
    return: The image with the triangles rendered.
    """

    # Calculate normals
    normals = calculate_normals(verts, faces.T)
    # Calculate barycentric coordinates
    bcoords = calculate_bcoords(verts, faces.T)

    # Project 3D points to 2D camera coordinate system
    p2d, depth = CameraLookingAt(focal, eye, lookat, up, verts)

    # Convert the 2D camera coordinate points to their corresponding pixel in the camera.
    p2d = rasterize(p2d, M, N, H, W)

    # Render the points from the camera's perspective
    I = render(p2d.T, normals, faces, vert_colors, bcoords, eye, mat, lights, light_amb, depth, bg_color, shader, light_type)

    return I
