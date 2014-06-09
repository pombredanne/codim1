import numpy as np
import matplotlib.pyplot as plt

from codim1.core import *
from codim1.assembly import *
from codim1.fast_lib import *
from codim1.post import *

# This file solves a Neumann problem using the hypersingular boundary
# integral equation -- also known as the traction boundary integral
# equation, the boundary conditions are tractions

def run(n_elements):
    degree = 2
    mesh = simple_line_mesh(50, left_edge = (-1.0, 0.0),
                                        right_edge = (1.0, 0.0))
    bf = basis_from_degree(degree)
    qs = QuadStrategy(mesh, 4, 12, 12, 12)
    ek = ElasticKernelSet(1.0, 0.25)
    def fnc(x, n):
        return [1.0, 0.0]

    apply_to_elements(mesh, "basis", bf, non_gen = True)
    apply_to_elements(mesh, "continuous", True, non_gen = True)
    apply_to_elements(mesh, "qs", qs, non_gen = True)
    init_dofs(mesh)
    apply_bc_from_fnc(mesh, fnc, "crack_traction")

    matrix, rhs = sgbem_assemble(mesh, ek)
    pin_ends_constraint(matrix, rhs, mesh, (0.0, 0.0), (0.0, 0.0))

    soln_coeffs = np.linalg.solve(matrix, rhs)

    x, u, t = evaluate_boundary_solution(mesh, soln_coeffs, 10)
    return x, u

def test_constant_traction_crack():
    n_elements = 40
    x, u = run(n_elements)
    correct = 0.75 * np.sqrt(1.0 - x[0, :] ** 2)
    error = np.sqrt(np.sum((u[0, :] - correct) ** 2 * (2.0 / n_elements)))
    assert(error <= 0.016)

# def plot():
    # plt.figure(1)
    # def plot_ux():
    #     plt.plot(x[0, :], u[0, :])
    #     plt.xlabel(r'X')
    #     plt.ylabel(r'$u_x$', fontsize = 18)
    # def plot_uy():
    #     plt.plot(x[0, :], u[1, :])
    #     plt.xlabel(r'X')
    #     plt.ylabel(r'$u_y$', fontsize = 18)
    # plot_ux()
    # plt.plot(x[0, :], correct)
    # plt.figure()
    # plot_uy()
    # plt.show()

# Compute some interior values.
# interior_stresses((-5, -5), (5, 5), 31, mesh, dh, qs, k_ta, k_h)
# x_pts = 31
# y_pts = 31
# x = np.linspace(-5, 5, x_pts)
# # Doesn't sample 0.0!
# y = np.linspace(-5, 5, y_pts)
# int_ux = np.zeros((x_pts, y_pts))
# int_uy = np.zeros((x_pts, y_pts))
# for i in range(x_pts):
#     for j in range(y_pts):
#         # traction_effect = 0.0 * ip.compute((x[i], y[j]), np.array((0.0, 0.0)),
#         #            k_d, traction_func)
#         displacement_effect = interior_pt(mesh, ((x[i], y[j]),
#                                           np.array([0.0, 0.0]))
#                                           k_t, soln)
#         int_ux[j, i] = traction_effect[0] + displacement_effect[0]
#         int_uy[j, i] = traction_effect[1] + displacement_effect[1]
#
# X, Y = np.meshgrid(x, y)
# plt.figure(3)
# plt.title('Displacement field for a constant stress drop crack.')
# plt.quiver(X, Y, int_ux, int_uy)
# plt.figure()
# plt.imshow(np.fliplr(int_ux))
# plt.colorbar()
# plt.figure()
# plt.imshow(np.fliplr(int_uy))
# plt.colorbar()
#
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
