import numpy as np
import matplotlib.pyplot as plt

from codim1.assembly import *
from codim1.core import *
import codim1.core.tools as tools
from codim1.fast_lib import AdjointTractionKernel,\
                                       RegularizedHypersingularKernel,\
                                       DisplacementKernel,\
                                       TractionKernel

# This file solves a Neumann problem using the hypersingular boundary
# integral equation -- also known as the traction boundary integral
# equation, the boundary conditions are tractions

# Elastic parameters
shear_modulus = 1.0
poisson_ratio = 0.25

degree = 3
n_elements = 50

# Quadrature points for the various circumstances
quad_min = degree + 1
quad_max = 12
quad_logr = 12
quad_oneoverr = 12
interior_quad_pts = 8

k_d = DisplacementKernel(shear_modulus, poisson_ratio)
k_t = TractionKernel(shear_modulus, poisson_ratio)
k_tp = AdjointTractionKernel(shear_modulus, poisson_ratio)
k_rh = RegularizedHypersingularKernel(shear_modulus, poisson_ratio)

left_end = np.array((-1.0, 1.0))
right_end = np.array((1.0, -1.0))
mesh = Mesh.simple_line_mesh(n_elements, left_end, right_end)
# tools.plot_mesh(mesh)
# plt.show()
bf = BasisFunctions.from_degree(degree)
qs = QuadStrategy(mesh, quad_min, quad_max, quad_logr, quad_oneoverr)
dh = DOFHandler(mesh, bf)
assembler = MatrixAssembler(mesh, bf, dh, qs)

# Build the rhs
src_strength = (right_end - left_end) /\
               np.linalg.norm(right_end - left_end)
def fnc(x, d):
    return src_strength[d]
    # return (4.0 / (np.pi * 3.0)) * ((1 / (x[0] - 1)) - (1 / (x[0] + 1)))
traction_func = BasisFunctions.from_function(fnc)
mass_matrix = MassMatrix(mesh, bf, traction_func, dh,
                         QuadGauss(degree + 1), compute_on_init = True)
rhs = -mass_matrix.for_rhs()

print('Assembling kernel matrix, Gpp')
# Use the basis function arclength derivatives for the regularized
# Gpp (hypersingular) kernel
# This is derived using integration by parts and moving part of the 1/r^3
# singularity onto the basis functions.

assembler = MatrixAssembler(mesh, bf, dh, qs)
derivs_assembler = MatrixAssembler(mesh, bf.get_gradient_basis(mesh), dh, qs)
Gpp = derivs_assembler.assemble_matrix(k_rh)

pin_ends_constraint(Gpp, rhs, (0.0, 0.0), (0.0, 0.0), dh)

soln_coeffs = np.linalg.solve(Gpp, rhs)

# Create a solution object that pairs the coefficients with the basis
soln = Solution(bf, dh, soln_coeffs)
x, s = tools.evaluate_boundary_solution(10, soln, mesh)

correct = 1.5 * np.sqrt(1.0 - x[:, 0] ** 2)
error = np.sqrt(np.sum((s[:, 0] - correct) ** 2 * (2.0 / n_elements)))
print error
print 2 * n_elements * bf.num_fncs
plt.figure(1)
def plot_ux():
    plt.plot(x[:, 0], s[:, 0])
    plt.xlabel(r'X')
    plt.ylabel(r'$u_x$', fontsize = 18)
def plot_uy():
    plt.plot(x[:, 0], s[:, 1])
    plt.xlabel(r'X')
    plt.ylabel(r'$u_y$', fontsize = 18)
plot_ux()
plt.plot(x[:, 0], correct)
plt.figure()
plot_uy()
plt.show()
# import sys
# sys.exit()

# Compute some interior values.
interior_stresses((-5, -5), (5, 5), 31, mesh, dh, qs, k_ta, k_h)
x_pts = 31
y_pts = 31
x = np.linspace(-5, 5, x_pts)
# Doesn't sample 0.0!
y = np.linspace(-5, 5, y_pts)
int_ux = np.zeros((x_pts, y_pts))
int_uy = np.zeros((x_pts, y_pts))
ip = InteriorPoint(mesh, dh, qs)
for i in range(x_pts):
    for j in range(y_pts):
        traction_effect = 0.0 * ip.compute((x[i], y[j]), np.array((0.0, 0.0)),
                   k_d, traction_func)
        displacement_effect = ip.compute((x[i], y[j]),
                                         np.array([0.0, 0.0]),
                                         k_t, soln)
        int_ux[j, i] = traction_effect[0] + displacement_effect[0]
        int_uy[j, i] = traction_effect[1] + displacement_effect[1]

X, Y = np.meshgrid(x, y)
plt.figure(3)
plt.title('Displacement field for a constant stress drop crack.')
plt.quiver(X, Y, int_ux, int_uy)
plt.figure()
plt.imshow(np.fliplr(int_ux))
plt.colorbar()
plt.figure()
plt.imshow(np.fliplr(int_uy))
plt.colorbar()

plt.xlabel('x')
plt.ylabel('y')
plt.show()
