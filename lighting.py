import numpy as np


def light(point, normal, vcolor, cam_pos, mat, lights, light_amb, light_type):
    """
    Calculates the light on a point

    param point: 3x1 vector of the point coordinates.
    param normal: 3x1 normal vector of the point.
    param vcolors: 3x1 color vector of the point.
    param cam_pos: 3x1 vector of the camera position coordinates.
    param mat: The PhongMaterial type of the object.
    param lights: A list of the point lights in the scene.
    return I: The intensity of the trichromatic radiation reflected from the point.
    """

    if light_type not in ["ambient", "diffuse", "specular", "all"]:
        print('ERROR: Invalid shading method "' + str(light_type) + '", choose from ["ambient", "diffuse", "specular", "all"]')
        exit(1)

    I = np.zeros((3, 1))

    # Calculate ambient light
    if light_type == 'ambient' or light_type == 'all':
        I += ambient_light(mat, light_amb)

    # Calculate diffuse light
    if light_type == 'diffuse' or light_type == 'all':
        I += diffuse_light(point, normal, mat, lights)

    # Calculate specular light
    if light_type == 'specular' or light_type == 'all':
        I += specular_light(point, normal, cam_pos, mat, lights)

    I *= vcolor
    return I


def ambient_light(mat, light_amb):
    """
    Calculates the ambient light

    param mat: The PhongMaterial type of the object.
    param lights: A list of the point lights in the scene.
    return I: The intensity of the trichromatic radiation reflected from the point.
    """

    return light_amb * mat.k_a


def diffuse_light(point, normal, mat, lights):
    """
    Calculates the diffuse light on a point

    param point: 3x1 vector of the point coordinates.
    param normal: 3x1 normal vector of the point.
    param mat: The PhongMaterial type of the object.
    param lights: A list of the point lights in the scene.
    return I: The intensity of the trichromatic radiation reflected from the point.
    """

    diffuse_light = np.zeros((3, 1))

    for light in lights:
        d = np.linalg.norm(light.pos.T - point)
        L = (light.pos.T - point) / d

        dotNL = max(np.dot(L.T[0], normal.T[0]), np.float64(0))
        diffuse_light += light.intensity.T * mat.k_d * dotNL

    return diffuse_light


def specular_light(point, normal, cam_pos, mat, lights):
    """
    Calculates the specular light on a point

    param point: 3x1 vector of the point coordinates.
    param normal: 3x1 normal vector of the point.
    param cam_pos: 3x1 vector of the camera position coordinates.
    param mat: The PhongMaterial type of the object.
    param lights: A list of the point lights in the scene.
    return I: The intensity of the trichromatic radiation reflected from the point.
    """

    specular_light = np.zeros((3, 1))

    V = (cam_pos - point) / np.linalg.norm(point - cam_pos)

    for light in lights:
        d = np.linalg.norm(light.pos.T - point)
        L = (light.pos.T - point) / d

        dotNL = np.dot(L.T[0], normal.T[0])
        R = (2 * normal * dotNL - L)
        R /= np.linalg.norm(R)
        specular_light += light.intensity.T * mat.k_s * np.dot(R.T[0], V.T[0]) ** mat.n_phong

    return specular_light
