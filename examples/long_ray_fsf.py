import numpy as np
import matplotlib.pyplot as plt
from codim1.core import *
from codim1.assembly import *
from codim1.fast_lib import *
from codim1.post import *
import codim1.core.tools as tools
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2

def test_long_ray_fsf():
    shear_modulus = 1.0
    poisson_ratio = 0.25
    n_elements_surface = 20
    # n_elements_surface = 25
    degree = 3
    quad_min = degree + 1
    quad_mult = 3
    quad_max = quad_mult * degree
    quad_logr = quad_mult * degree + (degree % 2)
    quad_oneoverr = quad_mult * degree + (degree % 2)
    interior_quad_pts = 13


    di = 1.0
    df = 1.0
    x_di = -0.5
    x_df = 0.5

    # Determine fault parameters
    # fault angle
    left_end = np.array((x_di, -di))
    right_end = np.array((x_df, -df))
    fault_vector = left_end - right_end
    # fault tangent and normal vectors
    fault_tangential = fault_vector / np.linalg.norm(fault_vector)
    fault_normal = np.array((fault_tangential[1], -fault_tangential[0]))

    # Mesh the surface
    main_surface_left = (-10.0, 0.0)
    main_surface_right = (10.0, 0.0)
    mesh1 = simple_line_mesh(n_elements_surface,
                            main_surface_left,
                            main_surface_right)

    per_step = 5
    steps = 10
    ray_lengths = [1.0] * per_step
    for i in range(1, steps):
        ray_lengths.extend([2.0 ** float(i)] * per_step)

    ray_left_dir = (-1.0, 0.0)
    mesh2 = ray_mesh(main_surface_left, ray_left_dir, ray_lengths, flip = True)
    ray_right_dir = (1.0, 0.0)
    mesh3 = ray_mesh(main_surface_right, ray_right_dir, ray_lengths)
    surface_mesh = combine_meshes(mesh2, combine_meshes(mesh1, mesh3),
                          ensure_continuity = True)
    apply_to_elements(surface_mesh, "bc",
                    BC("traction", ZeroBasis()), non_gen = True)

    # Mesh the fault
    fault_elements = 20
    fault_mesh = simple_line_mesh(fault_elements, left_end, right_end)
    apply_to_elements(fault_mesh, "bc", BC("crack_displacement",
                                     ConstantBasis(-fault_tangential)),
                                     non_gen = True)

    # Combine and apply pieces
    mesh = combine_meshes(surface_mesh, fault_mesh)

    bf = gll_basis(degree)
    qs = QuadStrategy(mesh, quad_min, quad_max, quad_logr, quad_oneoverr)
    apply_to_elements(mesh, "qs", qs, non_gen = True)
    apply_to_elements(mesh, "basis", bf, non_gen = True)
    apply_to_elements(mesh, "continuous", True, non_gen = True)
    init_dofs(mesh)

    ek = ElasticKernelSet(shear_modulus, poisson_ratio)

    matrix, rhs = sgbem_assemble(mesh, ek)
    apply_average_constraint(matrix, rhs, surface_mesh)
    # for e_k in surface_mesh:
    #     e_k.dofs_initialized = False
    # init_dofs(surface_mesh)
    # matrix2 = simple_matrix_assemble(surface_mesh, ek.k_rh)


    # The matrix produced by the hypersingular kernel is singular, so I need
    # to provide some further constraint in order to remove rigid body motions.
    # I impose a constraint that forces the average displacement to be zero.
    # apply_average_constraint(matrix, rhs, mesh)
    soln_coeffs = np.linalg.solve(matrix, rhs)

    x, u, t = evaluate_boundary_solution(surface_mesh, soln_coeffs, 8)


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

    # Compute the exact solution
    x_e = x[0, :]
    delta = np.arctan((df - di) / (x_df - x_di))
    ux_exact1, uy_exact1 = analytical_free_surface(x_e, x_di, di, delta, -1.0)
    ux_exact2, uy_exact2 = analytical_free_surface(x_e, x_df, df, delta, 1.0)
    ux_exact = ux_exact1 + ux_exact2
    uy_exact = uy_exact1 + uy_exact2

    def comparison_plot():
        plt.plot(x_e, ux_exact, '*', label = 'Exact X Displacement')
        plt.plot(x_e, uy_exact, '*', label = 'Exact Y Displacement')
        plt.plot(x_e, u[0, :], '8',
                 linewidth = 2, label = 'Estimated X displacement')
        plt.plot(x_e, u[1, :], '8',
                 linewidth = 2, label = 'Estimated Y displacement')
        plt.axis([-5, 5, -0.2, 0.2])
        plt.xlabel(r'$x/d$', fontsize = 18)
        plt.ylabel(r'$u/s$', fontsize = 18)
        plt.legend()
        plt.show()


    def error_plot():
        x_error = np.abs(ux_exact - u[0, :]) / np.abs(ux_exact)
        y_error = np.abs(uy_exact - u[1, :]) / np.abs(uy_exact)
        plt.figure(1)
        plt.xlim(-30, 30)
        plt.ylim(0, 0.0001)
        plt.plot(x_e, x_error, '*', label = '% X displacement Error')
        plt.plot(x_e, y_error, '*', label = '% Y displacement Error')
        plt.xlabel(r'$x/d$', fontsize = 18)
        plt.ylabel(r'$100\left(\frac{|u_{exact} - u_{est}|}{s}\right)$', fontsize = 18)
        plt.legend()
        plt.show()

    def interior_plot():
        x_pts = 30
        y_pts = 30
        min_x = -3
        max_x = 3
        min_y = -3
        max_y = 0
        x = np.linspace(min_x, max_x, x_pts)
        y = np.linspace(min_y, max_y, y_pts)
        int_ux = np.zeros((y_pts, x_pts))
        int_uy = np.zeros((y_pts, x_pts))
        for i in range(x_pts):
            print i
            for j in range(y_pts):
                u = sgbem_interior(mesh, (x[i], y[j]), np.zeros(2), ek, "displacement")
                int_ux[j, i] = u[0]
                int_uy[j, i] = u[1]

        X, Y = np.meshgrid(x, y)
        def contf_plot(type, data):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            levels = np.linspace(-0.5, 0.5, 21)
            tools.plot_mesh(fault_mesh, show = False, fig_ax = (fig, ax))
            im = ax.contourf(X, Y, data, levels)
            ax.contour(X, Y, data, levels, colors = ('k',), linestyles=['solid'])
            ax.set_ylabel(r'$x/d$', fontsize = 18)
            ax.set_xlabel(r'$y/d$', fontsize = 18)
            ax.set_title(type + ' displacement contours.')
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            fig.colorbar(im)
        contf_plot('Vertical', int_uy)
        contf_plot('Horizontal', int_ux)
        plt.show()

    # comparison_plot()

    # error_plot()
    # Forming interior plot
    interior_plot()


if __name__ == "__main__":
    test_long_ray_fsf()
