import numpy as np
import matplotlib.pyplot as plt
from codim1.core.dof_handler import ContinuousDOFHandler
from codim1.core.mesh import Mesh
from codim1.core.matrix_assembler import MatrixAssembler
from codim1.core.rhs_assembler import RHSAssembler
from codim1.core.basis_funcs import BasisFunctions
from codim1.fast.elastic_kernel import AdjointTractionKernel,\
                                       HypersingularKernel
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

mesh = Mesh.simple_line_mesh(100)
bf = BasisFunctions.from_degree(1, mesh)
qs = QuadStrategy(mesh, quad_min, quad_max, quad_logr, quad_oneoverr)
k_tp = AdjointTractionKernel(shear_modulus, poisson_ratio)
k_h = HypersingularKernel(shear_modulus, poisson_ratio)
dh = ContinuousDOFHandler(mesh, 1)
assembler = MatrixAssembler(mesh, bf, dh, qs)
# We used the basis function derivatives for the regularized Gpp kernel
# This is derived using integration by parts and moving part of the 1/r^3
# singularity onto the basis functions.
# TODO: This construct could be extended to handle regularizations of the
# cauchy singular kernels by allowing different sets of basis functions
# for the source and solution in the assembler.
derivs_assembler = MatrixAssembler(mesh, bf.derivs, dh, qs)
mass_matrix = MassMatrix(mesh, bf, dh, QuadGauss(2),
                         compute_on_init = True)

print('Assembling kernel matrix, Gpu')
Gpu = assembler.assemble_matrix(k_tp)
Gpu += 0.5 * mass_matrix.M
print('Assembling kernel matrix, Gpp')
Gpp = derivs_assembler.assemble_matrix(k_h)

fnc = lambda x: np.array([1.0, 0.0])
tractions = tools.interpolate(lambda x,n: fnc(x), dh, bf, mesh)
rhs_assembler = RHSAssembler(mesh, bf, dh, qs)
rhs = rhs_assembler.assemble_rhs(
        BasisFunctions.from_function(fnc), k_tp)
rhs = np.dot(Gpu, tractions)
# import ipdb; ipdb.set_trace()
soln = np.linalg.solve(Gpp, rhs)
x, s = tools.evaluate_boundary_solution(5, soln, dh, bf, mesh)

plt.figure(2)
plt.plot(x[:, 0], s[:, 0])
plt.xlabel(r'X')
plt.ylabel(r'$u_x$', fontsize = 18)
plt.figure(3)
plt.plot(x[:, 0], s[:, 1])
plt.xlabel(r'X')
plt.ylabel(r'$u_y$', fontsize = 18)
plt.show()
