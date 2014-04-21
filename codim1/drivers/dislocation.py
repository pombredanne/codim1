import cPickle
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from codim1.core import *
from codim1.assembly import *
from codim1.fast_lib import AdjointTractionKernel,\
                                       RegularizedHypersingularKernel,\
                                       DisplacementKernel,\
                                       TractionKernel,\
                                       HypersingularKernel
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
n_elements = 10
degree = 1

k_d = DisplacementKernel(shear_modulus, poisson_ratio)
k_t = TractionKernel(shear_modulus, poisson_ratio)
k_tp = AdjointTractionKernel(shear_modulus, poisson_ratio)
k_h = HypersingularKernel(shear_modulus, poisson_ratio)
k_rh = RegularizedHypersingularKernel(shear_modulus, poisson_ratio)

mesh = Mesh.simple_line_mesh(n_elements, -1.0, 1.0)
bf = BasisFunctions.from_degree(degree)
qs = QuadStrategy(mesh, quad_min, quad_max, quad_logr, quad_oneoverr)
dh = DOFHandler(mesh, bf, range(n_elements))

rhs_assembler = PointSourceRHS(mesh, bf.get_gradient_basis(mesh), dh, qs)
str_and_loc = [((1.0, 0.0), (-1.0, 0.0), (1.0, 0.0))]
              # ((1.0, 0.0), (1.0, 0.0), (1.0, 0.0))]
rhs = rhs_assembler.assemble_rhs(str_and_loc, k_rh)

def dd_fnc(x, d):
    if d == 1:
        return 0.0
    if x[0] <= -0.5 and x[0] >= -1:
        return 1.0
    return 0.0
dd_fnc_basis = BasisFunctions.from_function(dd_fnc)
rhs_assembler2 = RHSAssembler(mesh, bf, dh, qs)
rhs2 = rhs_assembler2.assemble_rhs(dd_fnc_basis, k_h)
import ipdb;ipdb.set_trace()

matrix = MassMatrix(mesh, bf, bf, dh, QuadGauss(degree + 1), True)
matrix = -matrix.M

soln_coeffs = np.linalg.solve(matrix, rhs2)


soln = Solution(bf, dh, soln_coeffs)
x, t = tools.evaluate_boundary_solution(8, soln, mesh)
x = x[25:]
t = t[25:]
plt.figure()
plt.plot(x[:, 0], t[:, 0], label = 'tx', linewidth = '3')
plt.plot(x[:, 0], t[:, 1], label = 'ty', linewidth = '3')

tx, ty = exact_edge_dislocation_trac(x[:, 0] + 1, x[:, 1], 0.0, 1.0)
tx2, ty2 = exact_edge_dislocation_trac(x[:, 0] - 1, x[:, 1], 0.0, 1.0)
exact_tx = tx# + tx
exact_ty = ty# + ty
plt.plot(x[:, 0], exact_tx, label = 'tx_exact', linewidth = '3')
plt.plot(x[:, 0], exact_ty, label = 'ty_exact', linewidth = '3')

plt.legend()
plt.show()
sys.exit()
