import sys
import numpy as np
import matplotlib.pyplot as plt
from codim1.core import *
from codim1.assembly import *
from codim1.fast_lib import *
import codim1.core.tools as tools


shear_modulus = 1.0
poisson_ratio = 0.25
offset = 1.0
x_pts = 30
y_pts = 30
n_elements = 80
degree = 3
quad_min = degree + 1
quad_max = 3 * degree
quad_logr = 3 * degree + 1
quad_oneoverr = 3 * degree + 1
interior_quad_pts = 13

ek = ElasticKernelSet(shear_modulus, poisson_ratio)

left_end = np.array((-1.0, 0.0))
right_end = np.array((1.0, -0.0))
mesh = simple_line_mesh(n_elements, left_end, right_end)
# tools.plot_mesh(mesh)
# plt.show()
bf = basis_from_degree(degree)
qs = QuadStrategy(mesh, quad_min, quad_max, quad_logr, quad_oneoverr)
apply_to_elements(mesh, "basis", bf, non_gen = True)
apply_to_elements(mesh, "continuous", False, non_gen = True)
apply_to_elements(mesh, "qs", qs, non_gen = True)
init_dofs(mesh)

def point_src(pt, normal):
    src_strength = (right_end - left_end) /\
                   np.linalg.norm(right_end - left_end)
    src_normal = np.array([0.0, 1.0])
    stress = np.array(k_sh.call(pt - left_end, src_normal, normal))
    traction = -2 * stress.dot(src_strength)
    # src_strength2 = -src_strength
    # src_normal2 = np.array([0.0, 1.0])
    # stress2 = np.array(k_sh.call(pt - right_end, src_normal2, normal))
    # traction -= 2 * stress2.dot(src_strength2)
    return traction

soln_coeffs = tools.interpolate(point_src, dh, bf, mesh)
# Get rid of the NANs at the tips
soln_coeffs = np.where(np.isnan(soln_coeffs), 0, soln_coeffs)

soln = Solution(bf, dh, soln_coeffs)
x, t = tools.evaluate_boundary_solution(8, soln, mesh)
tx = t[:, 0]
ty = t[:, 1]
distance_to_left = np.sqrt((x[:, 0] - left_end[0]) ** 2 +
                           (x[:, 1] - left_end[1]) ** 2)

x = np.linspace(-5, 5, x_pts)
# Doesn't sample 0.0!
y = np.linspace(-15, 15, y_pts)
X, Y = np.meshgrid(x, y)

x = np.linspace(-5, 5, x_pts)
# Doesn't sample 0.0!
y = np.linspace(-5, 5, y_pts)
sxx = np.zeros((x_pts, y_pts))
sxy = np.zeros((x_pts, y_pts))
sxy2 = np.zeros((x_pts, y_pts))
syy = np.zeros((x_pts, y_pts))
displacement = np.zeros((x_pts, y_pts, 2))
def fnc(x,d):
    if d == 0 and x[0] <= 1.0 and x[0] >= -1.0:
        return 1.0
    return 0.0
displacement_func = BasisFunctions.from_function(fnc)

ip = InteriorPoint(mesh, dh, qs)
for i in range(x_pts):
    print i
    for j in range(y_pts):
        displacement[j, i, :] += ip.compute((x[i], y[j]),
                                           np.array([0.0, 0.0]),
                                           k_t, displacement_func)
        # sxx[j, i], sxy[j, i] = 0.5 * point_src(np.array(x[i], y[j]),
        #                                        np.array((0.0, 1.0)))
        # sxy2[j, i], syy[j, i] = 0.5 * point_src(np.array(x[i], y[j]),
        #                                        np.array((1.0, 0.0)))
int_ux = displacement[:, :, 0]
int_uy = displacement[:, :, 1]

plt.figure(7)
plt.imshow(int_ux)
plt.title(r'Derived $u_x$')
plt.colorbar()

plt.figure(8)
plt.imshow(int_uy)
plt.title(r'Derived $u_y$')
plt.colorbar()

# plt.figure(9)
# plt.imshow(sxy)
# plt.title(r'Derived $s_{xy}$')
# plt.colorbar()
#
# plt.figure(10)
# plt.imshow(sxx)
# plt.title(r'Derived $s_{xx}$')
# plt.colorbar()
# plt.figure(11)
# plt.imshow(sxy2)
# plt.title(r'Derived $s_{xy2}$')
# plt.colorbar()
#
# plt.figure(12)
# plt.imshow(syy)
# plt.title(r'Derived $s_{yy}$')
# plt.colorbar()
# plt.figure(11)
# plt.imshow(int_ux - exact_grid_ux)
# plt.title(r'Error in $u_x$')
# plt.colorbar()

plt.show()
