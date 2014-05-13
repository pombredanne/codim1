import numpy as np
import matplotlib.pyplot as plt

import codim1.core.tools as tools
from codim1.core import *
from codim1.assembly import *
from codim1.fast_lib import *

shear_modulus = 1.0
poisson_ratio = 0.25
n_elements = 100
degree = 1
quad_min = 4
quad_max = 10
quad_logr = 10
quad_oneoverr = 10

k_d = DisplacementKernel(shear_modulus, poisson_ratio)
k_t = TractionKernel(shear_modulus, poisson_ratio)
k_ta = AdjointTractionKernel(shear_modulus, poisson_ratio)
k_rh = RegularizedHypersingularKernel(shear_modulus, poisson_ratio)
k_h = HypersingularKernel(shear_modulus, poisson_ratio)

bf = BasisFunctions.from_degree(degree)
mesh = circular_mesh(n_elements, 1.0)
qs = QuadStrategy(mesh, quad_max, quad_max, quad_logr, quad_oneoverr)
dh = DOFHandler(mesh, bf)#, range(n_elements))

matrix_assembler = MatrixAssembler(mesh, bf, dh, qs)
matrix = matrix_assembler.assemble_matrix(k_d)

# Uniform compression displacement
def compress(x, d):
    x_length = np.sqrt(x[0] ** 2 + x[1] ** 2)
    return 0.2 * x[d] / x_length

displacement_function = BasisFunctions.from_function(compress)

# Assemble the rhs, composed of the displacements induced by the
# traction inputs.
print("Assembling RHS")
rhs_assembler = RHSAssembler(mesh, bf, dh, qs)
rhs = rhs_assembler.assemble_rhs(displacement_function, k_t)

mass_matrix = MassMatrix(mesh, bf, displacement_function,
                         dh, QuadGauss(degree + 1),
                         compute_on_init = True)
rhs += mass_matrix.for_rhs()

soln_coeffs = np.linalg.solve(matrix, rhs)
soln = Solution(bf, dh, soln_coeffs)

# Evaluate that solution at 400 points around the circle
x, t = tools.evaluate_boundary_solution(400 / n_elements, soln, mesh)

plt.figure(2)
plt.plot(x[:, 0], t[:, 0])
plt.xlabel(r'X')
plt.ylabel(r'$t_x$', fontsize = 18)

plt.figure(3)
plt.plot(x[:, 0], t[:, 1])
plt.xlabel(r'X')
plt.ylabel(r'$t_y$', fontsize = 18)
plt.show()
