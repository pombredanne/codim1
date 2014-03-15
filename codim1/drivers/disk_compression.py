import time
import cPickle
import numpy as np
import matplotlib.pyplot as plt
from codim1.core.dof_handler import DiscontinuousDOFHandler,\
                                    ContinuousDOFHandler
from codim1.core.mesh import Mesh
from codim1.core.assembler import Assembler
from codim1.core.basis_funcs import BasisFunctions
from codim1.fast.elastic_kernel import ElastostaticKernel
from codim1.core.quad_strategy import QuadStrategy
from codim1.core.quadrature import QuadGauss
from codim1.core.interior_point import InteriorPoint
import codim1.core.tools as tools

def main(n_elements, element_deg, plot, interior_quad_pts):
    # Load from pickled matrix files.
    load = False

    # Elastic parameters
    shear_modulus = 1.0
    poisson_ratio = 0.25

    # Quadrature points for the various circumstances
    quad_min = 4
    quad_max = 12
    quad_logr = 12
    quad_oneoverr = 12


    #TODO: Implement isoparametric elements
    bf = BasisFunctions.from_degree(element_deg)
    # mesh = Mesh.simple_line_mesh(n_elements, -2.0, 2.0)

    # Is this an interior or exterior mesh?
    mesh = Mesh.circular_mesh(n_elements, 1.0)

    qs = QuadStrategy(mesh, quad_min, quad_max, quad_logr, quad_oneoverr)

    # tools.plot_mesh(mesh)
    kernel = ElastostaticKernel(shear_modulus, poisson_ratio)

    dh = ContinuousDOFHandler(mesh, element_deg)
    # dh = DiscontinuousDOFHandler(mesh, element_deg)
    assembler = Assembler(mesh, bf, kernel, dh, qs)

    if load:
        with open('H.matrix', 'rb') as f:
            H = cPickle.load(f)
        with open('G.matrix', 'rb') as f:
            G = cPickle.load(f)
    else:
        H, G = assembler.assemble()
        with open('H.matrix', 'wb') as f:
            cPickle.dump(H, f)
        with open('G.matrix', 'wb') as f:
            cPickle.dump(G, f)



    def full_surface_traction(x):
        return (x[0], x[1])

    def section_traction(x):
        if np.abs(x[0]) < np.cos(24 * (np.pi / 50)):
            return (-x[0], -x[1])
        return (0.0, 0.0)

    # G multiplies tractions
    # H mutliplies displacements
    def solve_traction_problem(fnc):
        displacements = tools.interpolate(fnc, dh, bf, mesh)
        rhs = np.dot(H, displacments)
        soln = np.linalg.solve(G, rhs)
        return soln, displacements

    def solve_displacement_problem(fnc):
        tractions = tools.interpolate(fnc, dh, bf, mesh)
        rhs = np.dot(G, tractions)
        soln = np.linalg.solve(H, rhs)
        return soln, tractions

    disp, trac = solve_displacement_problem(section_traction)

    x, s = tools.evaluate_boundary_solution(400 / n_elements, disp,
                                                dh, bf, mesh)

    # Compute along the y axis
    y_vals = np.linspace(0, 1.0, 20)[:-1]

    ip = InteriorPoint(mesh, bf, kernel, dh,
                       QuadGauss(interior_quad_pts, 0.0, 1.0))
    int_strs = np.array(
            [ip.compute_stress((0.0, y), disp, trac) for y in y_vals])
    int_disp = np.array(
            [ip.compute_displacement((0.0, y), disp, trac) for y in y_vals])

    if plot:
        plt.figure(2)
        plt.plot(x[:, 0], s[:, 0])
        plt.xlabel(r'X')
        plt.ylabel(r'$u_x$', fontsize = 18)
        plt.figure(3)
        plt.plot(x[:, 0], s[:, 1])
        plt.xlabel(r'X')
        plt.ylabel(r'$u_y$', fontsize = 18)
        plt.figure(4)
        plt.plot(y_vals, int_strs[:, 0, 0])
        # plt.figure(5)
        plt.plot(y_vals, int_strs[:, 1, 1])
        plt.xlabel('distance along y axis from origin')
        plt.ylabel(r'$\sigma_{xx}$ and $\sigma_{yy}$')
        # plt.figure(6)
        # plt.plot(y_vals, int_disp[:, 0])
        # plt.figure(7)
        # plt.plot(y_vals, int_disp[:, 1])
        plt.show()
    return int_strs[:, 0, 0]
    # See section 2.7 of starfield and crouch for the standard formulas to
    # convert from plane strain to plane stress.

if __name__ == "__main__":
    # strs = []
    # for i in range(15):
    #     strs.append(main(100, 1, False, i + 2))

    strs = []

    main(100, 1, True, 13)
    # start = time.time()

    # end = time.time()
    # print("Time: " + str(end - start))

    # main(5)
    # s1 = main(10, 1, False)
    # s2 = main(20, 1, False)
    # s4 = main(40, 1, False)
    # s8 = main(80, 1, True)

    # rmse = lambda x, y: np.sqrt(np.sum((x - y) ** 2) / s8.shape[0])
    # print rmse(s1, s2)
    # print rmse(s2, s4)
    # print rmse(s4, s8)
