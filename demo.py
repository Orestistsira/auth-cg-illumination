import numpy as np
import matplotlib.pyplot as plt

from phong_material import PhongMaterial
from point_light import PointLight
from render_object import render_object

# Load data from file
data = np.load('h3.npy', allow_pickle=True)[()]
verts = data['verts']
vert_colors = data['vertex_colors'].T
faces = data['face_indices'].T
eye = data['cam_eye']
lookat = data['cam_lookat']
up = data['cam_up']
focal = data['focal']
k_a = data['ka']
k_d = data['kd']
k_s = data['ks']
n = data['n']
light_positions = data['light_positions']
light_intensities = data['light_intensities']
Ia = data['Ia'].T[0]
M = data['M']
N = data['N']
W = data['W']
H = data['H']
bg_color = data['bg_color']

# Initialize lights and material
num_of_lights = len(light_positions)
lights = np.empty(num_of_lights, dtype=PointLight)

mat = PhongMaterial(k_a, k_d, k_s, n)
for i in range(num_of_lights):
    lights[i] = PointLight(np.array([light_positions[i]]), np.array([light_intensities[i]]))

# Render object with gouraud shader
print('Rendering object with gouraud shading...')
shader = "gouraud"

# Render with ambient light type
light_type = "ambient"
I = render_object(shader, focal, eye, lookat, up, bg_color, M, N, H, W, verts, vert_colors, faces, mat, lights, Ia, light_type)
# Save as image
plt.imsave('gouraud-ambient.jpg', np.array(I[::-1]))

# Render with diffuse light type
light_type = "diffuse"
I = render_object(shader, focal, eye, lookat, up, bg_color, M, N, H, W, verts, vert_colors, faces, mat, lights, Ia, light_type)
# Save as image
plt.imsave('gouraud-diffuse.jpg', np.array(I[::-1]))

# Render with specular light type
light_type = "specular"
I = render_object(shader, focal, eye, lookat, up, bg_color, M, N, H, W, verts, vert_colors, faces, mat, lights, Ia, light_type)
# Save as image
plt.imsave('gouraud-specular.jpg', np.array(I[::-1]))

# Render with all light types
light_type = "all"
I = render_object(shader, focal, eye, lookat, up, bg_color, M, N, H, W, verts, vert_colors, faces, mat, lights, Ia, light_type)
# Save as image
plt.imsave('gouraud-all.jpg.jpg', np.array(I[::-1]))


# Render object with phong shader
print('Rendering object with phong shading...')
shader = "phong"

# Render with ambient light type
light_type = "ambient"
I = render_object(shader, focal, eye, lookat, up, bg_color, M, N, H, W, verts, vert_colors, faces, mat, lights, Ia, light_type)
# Save as image
plt.imsave('phong-ambient.jpg', np.array(I[::-1]))

# Render with diffuse light type
light_type = "diffuse"
I = render_object(shader, focal, eye, lookat, up, bg_color, M, N, H, W, verts, vert_colors, faces, mat, lights, Ia, light_type)
# Save as image
plt.imsave('phong-diffuse.jpg', np.array(I[::-1]))

# Render with specular light type
light_type = "specular"
I = render_object(shader, focal, eye, lookat, up, bg_color, M, N, H, W, verts, vert_colors, faces, mat, lights, Ia, light_type)
# Save as image
plt.imsave('phong-specular.jpg', np.array(I[::-1]))

# Render with all light types
light_type = "all"
I = render_object(shader, focal, eye, lookat, up, bg_color, M, N, H, W, verts, vert_colors, faces, mat, lights, Ia, light_type)
# Save as image
plt.imsave('phong-all.jpg.jpg', np.array(I[::-1]))
