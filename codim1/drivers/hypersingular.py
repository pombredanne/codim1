import numpy as np
import matplotlib.pyplot as plt
from codim1.core.dof_handler import ContinuousDOFHandler
from codim1.core.mesh import Mesh
from codim1.core.matrix_assembler import MatrixAssembler
from codim1.core.rhs_assembler import RHSAssembler
from codim1.core.basis_funcs import BasisFunctions, Solution
from codim1.fast.elastic_kernel import AdjointTractionKernel,\
                                       RegularizedHypersingularKernel,\
                                       DisplacementKernel,\
                                       TractionKernel
from codim1.core.quad_strategy import QuadStrategy
from codim1.core.quadrature import QuadGauss
from codim1.core.mass_matrix import MassMatrix
from codim1.core.interior_point import InteriorPoint
import codim1.core.tools as tools

# This file solves a Neumann problem using the hypersingular boundary
# integral equation -- also known as the traction boundary integral
# equation, because all the terms are in traction units.

# Elastic parameters
shear_modulus = 1.0
poisson_ratio = 0.25

# Quadrature points for the various circumstances
quad_min = 4
quad_max = 12
quad_logr = 12
quad_oneoverr = 12
interior_quad_pts = 8

n_elements = 50

k_d = DisplacementKernel(shear_modulus, poisson_ratio)
k_t = TractionKernel(shear_modulus, poisson_ratio)
k_tp = AdjointTractionKernel(shear_modulus, poisson_ratio)
k_h = RegularizedHypersingularKernel(shear_modulus, poisson_ratio)

mesh = Mesh.simple_line_mesh(n_elements)
bf = BasisFunctions.from_degree(1)
qs = QuadStrategy(mesh, quad_min, quad_max, quad_logr, quad_oneoverr)
dh = ContinuousDOFHandler(mesh, 1)
assembler = MatrixAssembler(mesh, bf, dh, qs)
# We used the basis function derivatives for the regularized Gpp kernel
# This is derived using integration by parts and moving part of the 1/r^3
# singularity onto the basis functions.
derivs_assembler = MatrixAssembler(mesh, bf.get_gradient_basis(mesh), dh, qs)
mass_matrix = MassMatrix(mesh, bf, dh, QuadGauss(2),
                         compute_on_init = True)

# print('Assembling kernel matrix, Gpu')
# Gpu = assembler.assemble_matrix(k_tp)
# Gpu += 0.5 * mass_matrix.M
print('Assembling kernel matrix, Gpp')
Gpp = derivs_assembler.assemble_matrix(k_h)

fnc = lambda x: np.array([1.0, 0.0])
traction_function = BasisFunctions.from_function(fnc)
rhs_assembler = RHSAssembler(mesh, bf, dh, qs)
rhs = rhs_assembler.assemble_rhs(traction_function, k_tp)
soln_coeffs = np.linalg.solve(Gpp, rhs)
# Create a solution object that pairs the coefficients with the basis
soln = Solution(bf, dh, soln_coeffs)
x, s = tools.evaluate_boundary_solution(5, soln, mesh)

# plt.figure(1)
# plt.plot(x[:, 0], s[:, 0])
# plt.xlabel(r'X')
# plt.ylabel(r'$u_x$', fontsize = 18)
# plt.figure(2)
# plt.plot(x[:, 0], s[:, 1])
# plt.xlabel(r'X')
# plt.ylabel(r'$u_y$', fontsize = 18)
# plt.show()

# Compute some interior values.
x_pts = 98
y_pts = 98
x = np.linspace(-5, 5, x_pts)
# Doesn't sample 0.0!
y = np.linspace(-5, 5, y_pts)
int_ux = np.zeros((x_pts, y_pts))
int_uy = np.zeros((x_pts, y_pts))
interior_quadrature = QuadGauss(interior_quad_pts)
ip = InteriorPoint(mesh, dh, interior_quadrature)
for i in range(x_pts):
    for j in range(y_pts):
        traction_effect = ip.compute((x[i], y[j]), np.array((0.0, 0.0)),
                   k_d, traction_function)
        displacement_effect = -ip.compute((x[i], y[j]), np.array([0.0, 0.0]),
                   k_t, soln)
        int_ux[j, i] = traction_effect[0] + displacement_effect[0]
        int_uy[j, i] = traction_effect[1] + displacement_effect[1]

plt.figure(3)
plt.title(r'$u_x$')
plt.imshow(int_ux)
plt.colorbar()
plt.figure(4)
plt.title(r'$u_y$')
plt.imshow(int_uy)
plt.colorbar()
plt.show()
import ipdb;ipdb.set_trace()
