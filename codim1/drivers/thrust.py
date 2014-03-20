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

# Elastic parameter
shear_modulus = 1.0
poisson_ratio = 0.25

# Quadrature points for the various circumstances
quad_min = 4
quad_max = 12
quad_logr = 12
quad_oneoverr = 12
# I did some experiments and
# 13 Quadrature points seems like it gives error like 1e-10, lower
# than necessary, but nice for testing other components
interior_quad_pts = 13

n_elements = 50

k_d = DisplacementKernel(shear_modulus, poisson_ratio)
k_t = TractionKernel(shear_modulus, poisson_ratio)
k_tp = AdjointTractionKernel(shear_modulus, poisson_ratio)
k_h = RegularizedHypersingularKernel(shear_modulus, poisson_ratio)

mesh = Mesh.simple_line_mesh(n_elements)
bf = BasisFunctions.from_degree(1)
qs = QuadStrategy(mesh, quad_min, quad_max, quad_logr, quad_oneoverr)
dh = ContinuousDOFHandler(mesh, 1)

print('Assembling kernel matrix, Guu')
matrix_assembler = MatrixAssembler(mesh, bf, dh, qs)
Guu = matrix_assembler.assemble_matrix(k_d)

print('Assembling kernel matrix, Gup')
# TODO: Allow "mass matrix" creation on the RHS. Doesn't matter here, because
# there are no discontinuities, but it is bad practice to use the matrix like
# this.
mass_matrix = MassMatrix(mesh, bf, dh, QuadGauss(2),
                         compute_on_init = True)
Gup = matrix_assembler.assemble_matrix(k_t)
Gup -= 0.5 * mass_matrix.M

displacement_function = lambda x, n: np.array([1.0, 0.0])
displacement_coeffs = tools.interpolate(displacement_function, dh, bf, mesh)
rhs = np.dot(Gup, displacement_coeffs)

# Solve Ax = b, where x are the coefficients over the solution basis
soln_coeffs = np.linalg.solve(Guu, rhs)

# Create a solution object that pairs the coefficients with the basis
soln = Solution(bf, dh, soln_coeffs)

# TODO: Extract this interior point computation to some tool function.
x_pts = 50
y_pts = 50
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
                   k_d, soln)
        displacement_effect = -ip.compute(
                (x[i], y[j]), np.array([0.0, 0.0]), k_t,
                BasisFunctions.from_function(
                    lambda x: displacement_function(x, 0)))
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
