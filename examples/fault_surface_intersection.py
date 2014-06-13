import numpy as np
import matplotlib.pyplot as plt
from codim1.core import *
from codim1.assembly import *
from codim1.fast_lib import *
from codim1.post import *
import codim1.core.tools as tools
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2

def test_fault_surface_intersection():
    shear_modulus = 1.0
    poisson_ratio = 0.25
    n_elements_surface = 100
    # n_elements_surface = 25
    degree = 2
    quad_min = degree + 1
    quad_mult = 3
    quad_max = quad_mult * degree
    quad_logr = quad_mult * degree + (degree % 2)
    quad_oneoverr = quad_mult * degree + (degree % 2)
    interior_quad_pts = 13


    di = 0.0
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
    main_surface_seg1_left = (-10.0, 0.0)
    main_surface_seg1_right = (-0.5, 0.0)
    main_surface_seg2_left = (-0.5, 0.0)
    main_surface_seg2_right = (10.0, 0.0)
    mesh1seg1 = simple_line_mesh(n_elements_surface / 2,
                            main_surface_seg1_left,
                            main_surface_seg1_right)
    mesh1seg2 = simple_line_mesh(n_elements_surface / 2,
                            main_surface_seg2_left,
                            main_surface_seg2_right)
    mesh1 = combine_meshes(mesh1seg1, mesh1seg2, ensure_continuity = True)

    per_step = 5
    steps = 10
    ray_lengths = [1.0] * per_step
    for i in range(1, steps):
        ray_lengths.extend([2.0 ** float(i)] * per_step)

    ray_left_dir = (-1.0, 0.0)
    mesh2 = ray_mesh(main_surface_seg1_left, ray_left_dir,
                     ray_lengths, flip = True)
    ray_right_dir = (1.0, 0.0)
    mesh3 = ray_mesh(main_surface_seg2_right, ray_right_dir, ray_lengths)
    part1 = combine_meshes(mesh1, mesh3, ensure_continuity = True)
    surface_mesh = combine_meshes(mesh2, part1, ensure_continuity = True)
    apply_to_elements(surface_mesh, "bc",
                    BC("traction", ZeroBasis()), non_gen = True)

    # Mesh the fault
    fault_elements = 50
    fault_mesh = simple_line_mesh(fault_elements, left_end, right_end)
    apply_to_elements(fault_mesh, "bc", BC("crack_displacement",
                                     ConstantBasis(-fault_tangential)),
                                     non_gen = True)

    # Combine and apply pieces
    mesh = combine_meshes(surface_mesh, fault_mesh, ensure_continuity = True)

    bf = gll_basis(degree)
    qs = QuadStrategy(mesh, quad_min, quad_max, quad_logr, quad_oneoverr)
    apply_to_elements(mesh, "qs", qs, non_gen = True)
    apply_to_elements(mesh, "basis", bf, non_gen = True)
    sgbem_dofs(mesh)

    ek = ElasticKernelSet(shear_modulus, poisson_ratio)

    matrix, rhs = sgbem_assemble(mesh, ek)
    lse = fault_mesh.elements[0].vertex1.connected_to[0]
    rse = fault_mesh.elements[0].vertex1.connected_to[1]
    constraint_dofx = lse.dofs[0, -1]
    constraint_dofy = lse.dofs[1, -1]
    other_dofx = rse.dofs[0, 0]
    other_dofy = rse.dofs[1, 0]
    matrix[constraint_dofx, :] = 0
    matrix[constraint_dofy, :] = 0
    rhs[constraint_dofx] = -1 / np.sqrt(2)
    rhs[constraint_dofy] = 1 / np.sqrt(2)
    matrix[constraint_dofx, constraint_dofx] = -1
    matrix[constraint_dofx, other_dofx] = 1
    matrix[constraint_dofy, constraint_dofy] = -1
    matrix[constraint_dofy, other_dofy] = 1
    # apply_average_constraint(matrix, rhs, surface_mesh)

    # The matrix produced by the hypersingular kernel is singular, so I need
    # to provide some further constraint in order to remove rigid body motions.
    # I impose a constraint that forces the average displacement to be zero.
    # apply_average_constraint(matrix, rhs, mesh)
    soln_coeffs = np.linalg.solve(matrix, rhs)
    apply_coeffs(mesh, soln_coeffs, "soln")

    x, u, t = evaluate_boundary_solution(surface_mesh, soln_coeffs, 8)


    def analytical_free_surface(x, x_d, d, delta, s):
        """
        Analytical solution for the surface displacements from an infinite
        buried edge dislocation. Add two of them with opposite slip to represent
        an infinitely long thrust/normal fault.
        Extracted from chapter 3 of Segall 2010.
        """
        if abs(d) <= 1e-5:
            return analytical_free_surface_intersect(x, x_d, delta, s)
        xsi = (x - x_d) / d
        factor = s / np.pi
        term1 = np.cos(delta) * np.arctan(xsi)
        term2 = (np.sin(delta) - xsi * np.cos(delta)) / (1 + xsi ** 2)
        ux = factor * (term1 + term2)
        term1 = np.sin(delta) * np.arctan(xsi)
        term2 = (np.cos(delta) + xsi * np.sin(delta)) / (1 + xsi ** 2)
        uy = -factor * (term1 + term2)
        return ux, uy

    def analytical_free_surface_intersect(x, x_d, delta, s):
        factor = s / np.pi
        ux = factor * np.cos(delta) * (np.pi / 2) * np.sign(x - x_d)
        uy = -factor * np.sin(delta) * (np.pi / 2) * np.sign(x - x_d)
        return ux, uy

    # Compute the exact solution
    x_e = x[0, :]
    delta = np.arctan((df - di) / (x_df - x_di))
    ux_exact1, uy_exact1 = analytical_free_surface(x_e, x_di, di, delta, -1.0)
    ux_exact2, uy_exact2 = analytical_free_surface(x_e, x_df, df, delta, 1.0)
    ux_exact = ux_exact1 + ux_exact2
    uy_exact = uy_exact1 + uy_exact2
    # assert(np.sum(np.abs(ux_exact - u[0,:])) < 0.1)

    def comparison_plot():
        u[0, :] -= np.mean(u[0, :])
        u[1, :] -= np.mean(u[1, :])
        plt.plot(x_e, ux_exact, '-o', label = 'Exact X Displacement')
        plt.plot(x_e, uy_exact, '-o', label = 'Exact Y Displacement')
        plt.plot(x_e, u[0, :], '-o',
                 linewidth = 2, label = 'Estimated X displacement')
        plt.plot(x_e, u[1, :], '-o',
                 linewidth = 2, label = 'Estimated Y displacement')
        plt.xlim(-4, 4)
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
                u = sgbem_interior(mesh, (x[i], y[j]), np.zeros(2),
                                   ek, "soln", "displacement")
                int_ux[j, i] = u[0]
                int_uy[j, i] = u[1]

        X, Y = np.meshgrid(x, y)
        def contf_plot(type, data):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            levels = np.linspace(-0.5, 0.5, 21)
            tools.plot_mesh(fault_mesh, show = False, fig_ax = (fig, ax))
            im = ax.contourf(X, Y, data, levels)
            ax.contour(X, Y, data, levels,
                       colors = ('k',), linestyles=['solid'])
            ax.set_ylabel(r'$x/d$', fontsize = 18)
            ax.set_xlabel(r'$y/d$', fontsize = 18)
            ax.set_title(type + ' displacement contours.')
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            fig.colorbar(im)
        contf_plot('Vertical', int_uy)
        contf_plot('Horizontal', int_ux)
        plt.show()

    comparison_plot()

    # error_plot()
    # Forming interior plot
    # interior_plot()


if __name__ == "__main__":
    test_fault_surface_intersection()
