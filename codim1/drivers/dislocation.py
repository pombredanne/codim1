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

k_d = DisplacementKernel(shear_modulus, poisson_ratio)
k_t = TractionKernel(shear_modulus, poisson_ratio)
k_tp = AdjointTractionKernel(shear_modulus, poisson_ratio)
k_h = HypersingularKernel(shear_modulus, poisson_ratio)
k_sh = SemiRegularizedHypersingularKernel(shear_modulus, poisson_ratio)
k_rh = RegularizedHypersingularKernel(shear_modulus, poisson_ratio)

left_end = np.array((-1.0, 0.0))
right_end = np.array((1.0, -0.0))
mesh = Mesh.simple_line_mesh(n_elements, left_end, right_end)
# tools.plot_mesh(mesh)
# plt.show()
bf = BasisFunctions.from_degree(degree)
qs = QuadStrategy(mesh, quad_min, quad_max, quad_logr, quad_oneoverr)
dh = DOFHandler(mesh, bf, range(n_elements))

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
# plt.figure(1)
# plt.plot(distance_to_left, tx, label = 'tx', linewidth = '3')
# plt.plot(distance_to_left, ty, label = 'ty', linewidth = '3')

def exact_edge_dislocation_disp(X, Y):
    # The analytic displacement fields due to an edge dislocation.
    # Swap X and Y from the eshelby solution.
    nu = poisson_ratio
    factor = (offset / (2 * np.pi))
    R = np.sqrt(X ** 2 + Y ** 2)
    ux = factor * (np.arctan(X / Y) + \
                   (1.0 / (2 * (1 - nu))) * (X * Y / (R ** 2)))
    uy = factor * ((((1 - 2 * nu) / (2 * (1 - nu))) * np.log(1.0 / R)) + \
                   (1.0 / (2 * (1 - nu))) * ((X ** 2) / (R ** 2)))
    return ux, uy

def exact_edge_dislocation_trac(X, Y, nx, ny):
    # Analytical traction field due to an edge dislocation on a surface with
    # normal n
    # Swap X and Y from the normally given solution.
    nu = poisson_ratio
    factor = 2 * shear_modulus * offset / (2 * np.pi * (1 - nu))
    denom = (Y ** 2 + X ** 2) ** 2
    sxx = -Y * (3 * X ** 2 + Y ** 2) * (factor / denom)
    syy = Y * (X ** 2 - Y ** 2) * (factor / denom)
    sxy = X * (X ** 2 - Y ** 2) * (factor / denom)
    tx = sxx * nx + sxy * ny
    ty = sxy * nx + syy * ny
    return tx, ty

exact_tx, exact_ty = \
    exact_edge_dislocation_trac(x[:, 0] + 1, x[:, 1], 0.0, 1.0)
exact_tx2, exact_ty2 = \
    exact_edge_dislocation_trac(x[:, 0] - 1, x[:, 1], 0.0, 1.0)
exact_tx -= exact_tx2
exact_ty -= exact_ty2
# plt.plot(distance_to_left, exact_tx, label = 'exact_tx', linewidth = '3')
# plt.plot(distance_to_left, exact_ty, label = 'exact_ty', linewidth = '3')
# plt.legend()

# plt.figure(2)
# error_x = np.abs((tx - exact_tx) / exact_tx)
# plt.plot(error_x * 100)
# plt.xlabel('x')
# plt.ylabel(r'$|t - t_{exact}|$')
# plt.title('Percent error')
# plt.show()

x = np.linspace(-5, 5, x_pts)
# Doesn't sample 0.0!
y = np.linspace(-5, 5, y_pts)
X, Y = np.meshgrid(x, y)

exact_grid_ux, exact_grid_uy = exact_edge_dislocation_disp(X + 1, Y)
exact_grid_ux2, exact_grid_uy2 = exact_edge_dislocation_disp(X - 1, Y)
exact_grid_ux = exact_grid_ux2 - exact_grid_ux
exact_grid_uy = exact_grid_uy2 - exact_grid_uy

exact_grid_tx, exact_grid_ty = exact_edge_dislocation_trac(X + 1, Y, 0.0, 1.0)
exact_grid_tx2, exact_grid_ty2 =\
        exact_edge_dislocation_trac(X - 1, Y, 0.0, 1.0)
exact_grid_tx = exact_grid_tx2 - exact_grid_tx
exact_grid_ty = exact_grid_ty2 - exact_grid_ty

plt.figure(3)
plt.imshow(exact_grid_ux)
plt.colorbar()
plt.title(r'Exact $u_x$')
#
plt.figure(4)
plt.imshow(exact_grid_uy)
plt.colorbar()
plt.title(r'Exact $u_y$')
#
# plt.figure(5)
# plt.imshow(exact_grid_tx)
# plt.colorbar()
# plt.title(r'Exact $t_x$')
#
# plt.figure(6)
# plt.imshow(exact_grid_ty)
# plt.colorbar()
# plt.title(r'Exact $t_y$')

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
# #
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
