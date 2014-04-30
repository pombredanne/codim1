import numpy as np
import matplotlib.pyplot as plt
from codim1.core import *
import codim1.core.tools as tools

shear_modulus = 1.0
poisson_ratio = 0.25
offset = 1.0
x_pts = 30
y_pts = 30
n_elements_fault = 50
n_elements_surface = 50
degree = 2
quad_min = degree + 1
quad_max = 3 * degree
quad_logr = 3 * degree
quad_oneoverr = 3 * degree
interior_quad_pts = 13

left_end = np.array((-1.0, 1.0))
right_end = np.array((1.0, -1.0))
mesh = simple_line_mesh(n_elements_fault, left_end, right_end)
fault_normal = mesh.get_normal(0, 0.0)

left_surface = np.array((-10.0, 1.5))
right_surface = np.array((10.0, 1.5))
mesh2 = simple_line_mesh(n_elements_surface, left_surface, right_surface)

mesh = combine_meshes(mesh, mesh2)
tools.plot_mesh(mesh)
plt.axis([-5, 5, -5, 5])
plt.xlabel('x')
plt.ylabel('y')
plt.text(-2.0, 1.6, 'Earth surface')
plt.text(-0.5, 0.6, 'Thrust fault', rotation=-40)
plt.show()

bf = BasisFunctions.from_degree(degree)
qs = QuadStrategy(mesh, quad_min, quad_max, quad_logr, quad_oneoverr)
dh = DOFHandler(mesh, bf, range(n_elements))

def point_src(pt, normal):
    src_strength = (right_end - left_end) /\
                   np.linalg.norm(right_end - left_end)
    src_normal = np.array([0.0, 1.0])
    stress = np.array(k_sh.call(pt - left_end, normal, src_normal))
    src_strength2 = -src_strength
    src_normal2 = np.array([0.0, 1.0])
    stress2 = np.array(k_sh.call(pt - right_end, normal, src_normal2))
    traction = -2 * stress.dot(src_strength)
    traction -= 2 * stress2.dot(src_strength2)
    return traction

str_and_loc = [(1.0, left_end, fault_normal), (-1.0, right_end, fault_normal)]
