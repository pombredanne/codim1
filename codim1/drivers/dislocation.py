import sys
import numpy as np
import matplotlib.pyplot as plt
from codim1.core import *
from codim1.assembly import *
from codim1.fast_lib import *
import codim1.core.tools as tools

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

shear_modulus = 1.0
poisson_ratio = 0.25
offset = 1.0
quad_min = 12
quad_max = 12
quad_logr = 12
quad_oneoverr = 12
interior_quad_pts = 13
x_pts = 30
y_pts = 30
n_elements = 50
degree = 2

k_d = DisplacementKernel(shear_modulus, poisson_ratio)
k_t = TractionKernel(shear_modulus, poisson_ratio)
k_tp = AdjointTractionKernel(shear_modulus, poisson_ratio)
k_h = HypersingularKernel(shear_modulus, poisson_ratio)
k_sh = SemiRegularizedHypersingularKernel(shear_modulus, poisson_ratio)
k_rh = RegularizedHypersingularKernel(shear_modulus, poisson_ratio)

mesh = Mesh.simple_line_mesh(n_elements, -1.0, 1.0)
bf = BasisFunctions.from_degree(degree)
qs = QuadStrategy(mesh, quad_min, quad_max, quad_logr, quad_oneoverr)
dh = DOFHandler(mesh, bf, range(n_elements))

def point_src(pt, normal):
    src_pt = np.array((-1.0, 0.0))
    src_normal = np.array([0.0, 1.0])
    src_strength = np.array((1.0, 0.0))
    stress = np.array(k_sh.call(src_pt - pt,
                    src_normal,
                    np.array([0.0, 1.0])))
    src_pt2 = np.array((1.0, 0.0))
    src_normal2 = np.array([0.0, 1.0])
    src_strength2 = np.array((-1.0, 0.0))
    stress2 = np.array(k_sh.call(src_pt2 - pt,
                    src_normal2,
                    np.array([0.0, 1.0])))
    traction = 2 * stress.dot(src_strength)
    traction -= 2 * stress2.dot(src_strength2)
    return traction

soln_coeffs = tools.interpolate(point_src, dh, bf, mesh)

soln = Solution(bf, dh, soln_coeffs)
x, t = tools.evaluate_boundary_solution(8, soln, mesh)
tx = t[:, 0]
ty = t[:, 1]
plt.figure()
plt.plot(x[:, 0], tx, label = 'tx', linewidth = '3')
plt.plot(x[:, 0], ty, label = 'ty', linewidth = '3')

exact_tx, exact_ty = \
    exact_edge_dislocation_trac(x[:, 0] + 1, x[:, 1], 0.0, 1.0)
exact_tx2, exact_ty2 = \
    exact_edge_dislocation_trac(x[:, 0] - 1, x[:, 1], 0.0, 1.0)
exact_tx += exact_tx2
exact_ty += exact_ty2
# plt.plot(x[:, 0], exact_tx, label = 'tx_exact', linewidth = '3')
# plt.plot(x[:, 0], exact_ty, label = 'ty_exact', linewidth = '3')
plt.legend()


plt.figure()
error_x = np.abs((tx - exact_tx) / exact_tx)
plt.plot(error_x * 100)
plt.xlabel('x')
plt.ylabel(r'$|t - t_{exact}|$')
plt.title('Percent error')
plt.show()


