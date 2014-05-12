import numpy as np
import matplotlib.pyplot as plt
from codim1.core import *
from codim1.assembly import *
from codim1.fast_lib import *
import codim1.core.tools as tools
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2

shear_modulus = 1.0
poisson_ratio = 0.25
n_elements_surface = 60
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

di = 0.1
df = 1.0
x_di = 0.0
x_df = 1.0
# fault angle
delta = np.arctan((df - di) / (x_df - x_di))
left_end = np.array((x_di, -di))
right_end = np.array((x_df, -df))
fault_vector = left_end - right_end
fault_tangential = fault_vector / np.linalg.norm(fault_vector)
fault_normal = np.array((fault_tangential[1], -fault_tangential[0]))

# left_surface = np.array((-10.0, 0.0))
# right_surface = np.array((10.0, 0.0))
# mesh = simple_line_mesh(n_elements_surface, left_surface, right_surface)

left_surface = np.array((-30.0, 0.0))
rise_begin = np.array((0.0, 0.0))
rise_end = np.array((3.0, 0.3))
right_surface = np.array((30.0, 0.3))
mesh1 = simple_line_mesh(n_elements_surface, left_surface, rise_begin)
mesh2 = simple_line_mesh(n_elements_surface, rise_begin, rise_end)
mesh3 = simple_line_mesh(n_elements_surface / 3, rise_end, right_surface)
mesh = combine_meshes(mesh1, combine_meshes(mesh2, mesh3),
                      ensure_continuity = True)
tools.plot_mesh(mesh)
plt.plot([x_di, x_df], [-di, -df], 'r-')
plt.axis([-1.0, 4.0, -2.5, 2.5])
plt.xlabel(r'$x/d$', fontsize = 18)
plt.ylabel(r'$y/d$', fontsize = 18)
plt.text(0.50, 0.25, r'slope $\approx$ 6 degrees', rotation = 5.0)
plt.text(-0.5, 0.05, 'flat')
plt.text(3.3, 0.35, 'flat')
plt.text(0.0, -0.4, 'Thrust fault (dip = 45 degrees)', rotation=-38)
plt.suptitle('Geometry for topographic rise problem', fontsize = 18)
plt.title('Total rise = 0.3 fault lengths.', fontsize = 16)
plt.show()

bf = BasisFunctions.from_degree(degree)
qs = QuadStrategy(mesh, quad_min, quad_max, quad_logr, quad_oneoverr)
dh = DOFHandler(mesh, bf)

str_and_loc = [(-fault_tangential, left_end, fault_normal),
               (fault_tangential, right_end, fault_normal)]
rhs_assembler = PointSourceRHS(mesh, bf.get_gradient_basis(mesh), dh, qs)
rhs = -rhs_assembler.assemble_rhs(str_and_loc, k_rh)

matrix_assembler = MatrixAssembler(mesh, bf.get_gradient_basis(mesh), dh, qs)
matrix = matrix_assembler.assemble_matrix(k_rh)

# The matrix produced by the hypersingular kernel is singular, so I need
# to provide some further constraint in order to remove rigid body motions.
# I impose a constraint that forces the average displacement to be zero.
apply_average_constraint(matrix, rhs, mesh, bf, dh)

soln_coeffs = np.linalg.solve(matrix, rhs)
soln = Solution(bf, dh, soln_coeffs)
x, u_soln = tools.evaluate_boundary_solution(4, soln, mesh)


def analytical_free_surface(x, x_d, d, delta, s):
    """
    Analytical solution for the surface displacements from an infinite
    buried edge dislocation. Add two of them with opposite slip to represent
    an infinitely long thrust/normal fault.
    Extracted from chapter 3 of Segall 2010.
    """
    xsi = (x - x_d) / d
    factor = s / np.pi
    term1 = np.cos(delta) * np.arctan(xsi)
    term2 = (np.sin(delta) - xsi * np.cos(delta)) / (1 + xsi ** 2)
    ux = factor * (term1 + term2)
    term1 = np.sin(delta) * np.arctan(xsi)
    term2 = (np.cos(delta) + xsi * np.sin(delta)) / (1 + xsi ** 2)
    uy = -factor * (term1 + term2)
    return ux, uy

plot_inner = 100
x_e = x[plot_inner:-plot_inner, 0]
ux_exact1, uy_exact1 = analytical_free_surface(x_e, x_di, di, delta, -1.0)
ux_exact2, uy_exact2 = analytical_free_surface(x_e, x_df, df, delta, 1.0)
ux_exact = ux_exact1 + ux_exact2
uy_exact = uy_exact1 + uy_exact2

plt.plot(x_e, ux_exact, '*', label = 'Exact X Displacement')
plt.plot(x_e, uy_exact, '*', label = 'Exact Y Displacement')
plt.plot(x[plot_inner:-plot_inner, 0],
         u_soln[plot_inner:-plot_inner, 0],
         linewidth = 2, label = 'Estimated X displacement')
plt.plot(x[plot_inner:-plot_inner, 0],
         u_soln[plot_inner:-plot_inner, 1],
         linewidth = 2, label = 'Estimated Y displacement')
plt.xlabel(r'$x/d$', fontsize = 18)
plt.ylabel(r'$u/s$', fontsize = 18)
plt.legend()
plt.show()
import ipdb;ipdb.set_trace()
