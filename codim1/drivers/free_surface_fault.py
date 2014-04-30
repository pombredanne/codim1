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
n_elements_surface = 100
degree = 2
quad_min = degree + 1
quad_mult = 5
quad_max = quad_mult * degree
quad_logr = quad_mult * degree + (degree % 2)
quad_oneoverr = quad_mult * degree + (degree % 2)
interior_quad_pts = 13

k_d = DisplacementKernel(shear_modulus, poisson_ratio)
k_t = TractionKernel(shear_modulus, poisson_ratio)
k_tp = AdjointTractionKernel(shear_modulus, poisson_ratio)
k_h = HypersingularKernel(shear_modulus, poisson_ratio)
k_sh = SemiRegularizedHypersingularKernel(shear_modulus, poisson_ratio)
k_rh = RegularizedHypersingularKernel(shear_modulus, poisson_ratio)

left_end = np.array((-1.0, -0.2))
right_end = np.array((1.0, -1.2))
mesh = simple_line_mesh(1, left_end, right_end)
fault_normal = mesh.get_normal(0, 0.0)
fault_tangential = np.array((fault_normal[1], -fault_normal[0]))

left_surface = np.array((-10.0, 0.0))
right_surface = np.array((10.0, 0.0))
mesh = simple_line_mesh(n_elements_surface, left_surface, right_surface)

bf = BasisFunctions.from_degree(degree)
qs = QuadStrategy(mesh, quad_min, quad_max, quad_logr, quad_oneoverr)
dh = DOFHandler(mesh, bf)

# def point_src(pt, normal):
#     src_strength = (right_end - left_end) /\
#                    np.linalg.norm(right_end - left_end)
#     src_normal = np.array([0.0, 1.0])
#     stress = np.array(k_sh.call(pt - left_end, normal, src_normal))
#     src_strength2 = -src_strength
#     src_normal2 = np.array([0.0, 1.0])
#     stress2 = np.array(k_sh.call(pt - right_end, normal, src_normal2))
#     traction = -2 * stress.dot(src_strength)
#     traction -= 2 * stress2.dot(src_strength2)
#     return traction

str_and_loc = [(fault_tangential, left_end, fault_normal),
               (-fault_tangential, right_end, fault_normal)]
rhs_assembler = PointSourceRHS(mesh, bf, dh, qs)
rhs = -rhs_assembler.assemble_rhs(str_and_loc, k_rh)

matrix_assembler = MatrixAssembler(mesh, bf, dh, qs)
matrix = matrix_assembler.assemble_matrix(k_rh)

soln_coeffs = np.linalg.solve(matrix, rhs)
soln = Solution(bf, dh, soln_coeffs)
x, s = tools.evaluate_boundary_solution(10, soln, mesh)
plt.plot(x[:, 0], s[:, 0], label = 'X displacement')
plt.plot(x[:, 0], s[:, 1], label = 'Y displacement')
plt.legend()
plt.show()
