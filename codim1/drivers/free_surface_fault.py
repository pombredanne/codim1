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
n_elements_surface = 200
degree = 1
quad_min = degree + 1
quad_mult = 3
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

di = -0.5
df = -1.5
x_di = 0.0
x_df = 1.0
# fault angle
delta = np.arctan((di - df) / (x_df - x_di))
left_end = np.array((x_di, di))
right_end = np.array((x_df, df))
mesh = simple_line_mesh(1, left_end, right_end)
fault_normal = mesh.get_normal(0, 0.0)
fault_tangential = np.array((-fault_normal[1], fault_normal[0]))

left_surface = np.array((-10.0, 0.0))
right_surface = np.array((10.0, 0.0))
mesh = simple_line_mesh(n_elements_surface, left_surface, right_surface)

bf = BasisFunctions.from_degree(degree)
qs = QuadStrategy(mesh, quad_min, quad_max, quad_logr, quad_oneoverr)
dh = DOFHandler(mesh, bf)

str_and_loc = [(-fault_tangential, left_end, fault_normal),
               (fault_tangential, right_end, fault_normal)]
rhs_assembler = PointSourceRHS(mesh, bf.get_gradient_basis(mesh), dh, qs)
rhs = -rhs_assembler.assemble_rhs(str_and_loc, k_rh)
plt.plot(rhs[dh.dof_map[0, 0, 0]:dh.dof_map[0, -1, -1]])
plt.plot(rhs[dh.dof_map[1, 0, 0]:dh.dof_map[1, -1, -1]], linewidth = 2)
plt.show()

matrix_assembler = MatrixAssembler(mesh, bf.get_gradient_basis(mesh), dh, qs)
matrix = matrix_assembler.assemble_matrix(k_rh)

# The matrix produced by the hypersingular kernel is singular, so I need
# to provide some further constraint in order to remove rigid body motions.
# I impose a constraint that forces the average displacement to be zero.
apply_average_constraint(matrix, rhs, mesh, bf, dh)
# pin_ends_constraint(matrix, rhs, (0.034, 0.0), (-0.057, 0.0), dh)

soln_coeffs = np.linalg.solve(matrix, rhs)
soln = Solution(bf, dh, soln_coeffs)
x, u_soln = tools.evaluate_boundary_solution(4, soln, mesh)
plt.plot(x[:, 0], u_soln[:, 0], label = 'X displacement')
plt.plot(x[:, 0], u_soln[:, 1], label = 'Y displacement')
plt.xlabel(r'$x/d$')
plt.ylabel(r'$u/s$')
plt.grid(True)

def analytical_free_surface(x, x_d, d, delta, s):
    xsi = (x - x_d) / d
    factor = s / np.pi
    term1 = np.cos(delta) * np.arctan(xsi)
    term2 = (np.sin(delta) - xsi * np.cos(delta)) / (1 + xsi ** 2)
    ux = factor * (term1 + term2)
    term1 = np.sin(delta) * np.arctan(xsi)
    term2 = (np.cos(delta) + xsi * np.sin(delta)) / (1 + xsi ** 2)
    uy = -factor * (term1 + term2)
    return ux, uy

x_e = x[:, 0]
ux_exact1, uy_exact1 = analytical_free_surface(x_e, x_di, di, delta, 1.0)
ux_exact2, uy_exact2 = analytical_free_surface(x_e, x_df, df, delta, -1.0)
ux_exact = ux_exact1 + ux_exact2
uy_exact = uy_exact1 + uy_exact2
plt.plot(x_e, ux_exact, '*-', label = 'Ux Exact')
plt.plot(x_e, uy_exact, '+-', label = 'Uy Exact')
plt.legend()
plt.show()
import ipdb;ipdb.set_trace()
