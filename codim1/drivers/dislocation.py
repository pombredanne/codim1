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

from matplotlib import rcParams
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
# rcParams['text.usetex'] = True

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
#
# def plot_edge_dislocation():
#     # Plot the solution for an edge dislocation with burgers
#     x_pts = 100
#     y_pts = 100
#     x = np.linspace(-5, 5, x_pts)
#     # Doesn't sample 0.0!
#     y = np.linspace(-5, 5, y_pts)
#     X, Y = np.meshgrid(x, y)
#
#     ux, uy = exact_edge_dislocation_disp(X + 1, Y)
#     ux2, uy2 = exact_edge_dislocation_disp(X - 1, Y)
#     ux = ux2 - ux
#     uy = uy2 - uy
#
#     tx, ty = exact_edge_dislocation_trac(X + 1, Y, 0.0, 1.0)
#     tx2, ty2 = exact_edge_dislocation_trac(X - 1, Y, 0.0, 1.0)
#     tx = tx2 - tx
#     ty = ty2 - ty
#
#     plt.figure()
#     plt.imshow(ux)
#     plt.colorbar()
#     plt.figure()
#     plt.imshow(uy)
#     plt.colorbar()
#     plt.figure()
#     plt.imshow(tx)
#     plt.colorbar()
#     plt.figure()
#     plt.imshow(ty)
#     plt.colorbar()
#     return ux, uy
#
#
# def reload_and_postprocess():
#     f = open('data/dislocation2/int_u.pkl', 'rb')
#     int_u = cPickle.load(f)
#     x = np.linspace(-5, 5, x_pts)
#     # Doesn't sample 0.0!
#     y = np.linspace(-5, 5, y_pts)
#     X, Y = np.meshgrid(x, y)
#
#     # Quiver plot
#     quiver_plot = plt.quiver(X[::2, ::2], Y[::2, ::2],
#                             int_u[0, ::2, ::2], int_u[1, ::2, ::2])
#     plt.quiverkey(quiver_plot, 0.60, 0.95, 0.25, r'0.25', labelpos='W')
#     plt.plot([-1, 1], [0, 0], 'r-', linewidth=4)
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.xlim([-5, 5])
#     plt.title(r'Displacement vectors for an edge dislocation ' +
#               'from $x = 1$ to $x = -1$.')
#
#     # Streamline plot
#     plt.figure()
#     quiver_plot = plt.streamplot(X, Y, int_u[0, :, :], int_u[1, :, :])
#     plt.plot([-1, 1], [0, 0], 'r-', linewidth=4)
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title(r'Displacement streamlines for an edge dislocation ' +
#               'from $x = 1$ to $x = -1$.')
#     plt.show()


# Elastic parameter
shear_modulus = 1.0
poisson_ratio = 0.25

# How far to move the dislocation
offset = 1.0

# Quadrature points for the various circumstances
quad_min = 4
quad_max = 12
quad_logr = 12
quad_oneoverr = 12
# I did some experiments and
# 13 Quadrature points seems like it gives error like 1e-10, lower
# than necessary, but nice for testing other components
interior_quad_pts = 13
# How many interior points to use.
x_pts = 30
y_pts = 30

n_elements = 50
degree = 1

# save = False

# if len(sys.argv) > 1 and sys.argv[1] == 'reload':
#     reload_and_postprocess()
#     sys.exit()

# x = np.linspace(-1, 1, 100)
# tx, ty = exact_edge_dislocation_trac(x + 1, 0.0 * x, 0.0, 1.0)
# tx2, ty2 = exact_edge_dislocation_trac(x - 1, 0.0 * x, 0.0, 1.0)
# tx -= tx2
# ty -= ty2
# plt.figure()
# plt.plot(x, tx)
# plt.figure()
# plt.plot(x, ty)
# plt.show()
# ux_exact, uy_exact = plot_edge_dislocation()
# plt.show()
# sys.exit()

start = time.time()

# The four kernels of linear elasticity!
# http://en.wikipedia.org/wiki/The_Three_Christs_of_Ypsilanti
k_d = DisplacementKernel(shear_modulus, poisson_ratio)
k_t = TractionKernel(shear_modulus, poisson_ratio)
k_tp = AdjointTractionKernel(shear_modulus, poisson_ratio)
k_h = HypersingularKernel(shear_modulus, poisson_ratio)
k_rh = RegularizedHypersingularKernel(shear_modulus, poisson_ratio)

mesh = Mesh.simple_line_mesh(n_elements)
bf = BasisFunctions.from_degree(degree)
qs = QuadStrategy(mesh, quad_min, quad_max, quad_logr, quad_oneoverr)
dh = DOFHandler(mesh, bf, range(n_elements))

print('Assembling RHS')
# The prescribed displacement discontinuity derivatve is 0.0 everywhere
# except for a point source at the tips of the dislocation
def fnc(x, d):
    if d is 1:
        return 0.0
    return 1.5 * np.sqrt(1 - x[0] ** 2)
def fnc2(x, n):
    return np.array([fnc(x, 0), fnc(x, 1)])
def fnc_deriv(x, d):
    if d is 1:
        return 0.0
    return -1.5 * x[0] / np.sqrt(1 - x[0] ** 2)
displacement_func = BasisFunctions.from_function(fnc)
displacement_func_deriv = BasisFunctions.from_function(fnc_deriv)


rhs_assembler = PointSourceRHS(mesh, bf.get_gradient_basis(mesh), dh, qs)
left_sing_pt = np.array([1.0, 0.0])
left_sing_n = np.array([0.0, 1.0])
right_sing_pt = np.array([-1.0, 0.0])
right_sing_n = np.array([0.0, 1.0])
singular_pts = [(1.0, left_sing_pt, left_sing_n)]
                # (1.0, right_sing_pt, right_sing_n)]
rhs = rhs_assembler.assemble_rhs(singular_pts, k_rh)
# derivs_assembler = MatrixAssembler(mesh, bf.get_gradient_basis(mesh), dh, qs)
# Gpp = derivs_assembler.assemble_matrix(k_rh)
# du = tools.interpolate(fnc2, dh, bf, mesh)
# rhs = Gpp.dot(du)
# rhs = np.array([ 0.  , -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02,
#        -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02,
#        -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02,
#        -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02,
#        -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02,
#        -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02,
#        -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02,
#        -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02,
#        -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02,
#        -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02,
#        -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02,
#        -0.02,  0.  ,  0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  ,
#        -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  ,
#        -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  ,
#        -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  ,
#        -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  ,
#        -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  ,
#        -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  ,
#        -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  ,
#        -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  ,
#        -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  ,
#        -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  ,
#        -0.  , -0.  , -0.  ,  0.  ])

print("Assembling Matrix")
mass_matrix = MassMatrix(mesh, bf, bf, dh, QuadGauss(degree + 1),
                         compute_on_init = True)
matrix = -mass_matrix.M

import ipdb;ipdb.set_trace()

print("Solving system")
# Solve Ax = b, where x are the coefficients over the solution basis
soln_coeffs = np.linalg.solve(matrix, rhs)

# Create a solution object that pairs the coefficients with the basis
soln = Solution(bf, dh, soln_coeffs)

# Compute some values of the solution along the boundary
x, t = tools.evaluate_boundary_solution(8, soln, mesh)
plt.figure()
plt.plot(x[:, 0], t[:, 0], label = 'tx', linewidth = '3')
plt.plot(x[:, 0], t[:, 1], label = 'ty', linewidth = '3')
# plt.show()
# sys.exit()
# Plot the exact value of the traction...
tx, ty = exact_edge_dislocation_trac(x[:, 0] + 1, x[:, 1], 0.0, 1.0)
tx2, ty2 = exact_edge_dislocation_trac(x[:, 0] - 1, x[:, 1], 0.0, 1.0)
exact_tx = tx2 - tx
exact_ty = ty2 - ty
print exact_ty
plt.plot(x[:, 0], exact_tx, label = 'tx_exact', linewidth = '3')
plt.plot(x[:, 0], exact_ty, label = 'ty_exact', linewidth = '3')

#
plt.legend()
#
plt.show()

# Calculate the error in the surface tractions.
# error_x = tools.L2_error(t[1:-1, 0], exact_tx[1:-1])
# error_y = tools.L2_error(t[1:-1, 1], exact_ty[1:-1])
# print error_x
# print error_y
# import ipdb;ipdb.set_trace()
#

print("Performing Interior Computation")
x = np.linspace(-5, 5, x_pts)
y = np.linspace(-5, 5, y_pts)
displacement_effect = np.zeros((x_pts, y_pts, 2))
ip = InteriorPoint(mesh, dh, qs)
for i in range(x_pts):
    for j in range(y_pts):
        displacement_effect[j, i, :] = \
            ip.compute((x[i], y[j]), np.array([0.0, 0.0]),
                            k_t, displacement_func)

int_ux = displacement_effect[:, :, 0]
int_uy = displacement_effect[:, :, 1]

plt.figure()
plt.imshow(int_ux)
plt.colorbar()
plt.figure()
plt.imshow(int_uy)
plt.colorbar()
plt.show()
