import time

import cPickle
import numpy as np
import matplotlib.pyplot as plt
from codim1.core.dof_handler import DiscontinuousDOFHandler,\
                                    ContinuousDOFHandler
from codim1.core.mesh import Mesh
from codim1.core.matrix_assembler import MatrixAssembler
from codim1.core.rhs_assembler import RHSAssembler
from codim1.core.basis_funcs import BasisFunctions
from codim1.fast.elastic_kernel import \
    DisplacementKernel, TractionKernel, \
    AdjointTractionVolumeKernel, HypersingularVolumeKernel
from codim1.core.quad_strategy import QuadStrategy
from codim1.core.quadrature import QuadGauss
from codim1.core.mass_matrix import MassMatrix
from codim1.core.interior_point import InteriorPoint
import codim1.core.tools as tools

def main(n_elements, element_deg, plot, interior_quad_pts, dof_type):
    # Load from pickled matrix files.
    load = False

    # Elastic parameters
    shear_modulus = 1.0
    poisson_ratio = 0.25

    # Quadrature points for the various circumstances
    quad_min = 4
    quad_max = 10
    quad_logr = 10
    quad_oneoverr = 10

    # Is this an interior or exterior mesh?
    mesh = Mesh.circular_mesh(n_elements, 1.0)

    #TODO: Implement isoparametric elements
    bf = BasisFunctions.from_degree(element_deg, mesh)

    qs = QuadStrategy(mesh, quad_min, quad_max, quad_logr, quad_oneoverr)

    # tools.plot_mesh(mesh)
    k_d = DisplacementKernel(shear_modulus, poisson_ratio)
    k_t = TractionKernel(shear_modulus, poisson_ratio)

    if dof_type == 'Disc':
        dh = DiscontinuousDOFHandler(mesh, element_deg)
    elif dof_type == 'Cont':
        dh = ContinuousDOFHandler(mesh, element_deg)

    matrix_assembler = MatrixAssembler(mesh, bf, dh, qs)
    rhs_assembler = RHSAssembler(mesh, bf, dh, qs)
    mass_matrix = MassMatrix(mesh, bf, dh, QuadGauss(element_deg + 1),
                             compute_on_init = True)

    # if load:
    #     with open('Guu.matrix', 'rb') as f:
    #         Guu = cPickle.load(f)
    #     with open('Gup.matrix', 'rb') as f:
    #         Gup = cPickle.load(f)
    # else:
    # We don't need Guu because it multiplies the RHS, which we can evaluate
    # directly from the function.
    # print('Assembling kernel matrix, Guu')
    # Guu = matrix_assembler.assemble_matrix(k_d)
    print('Assembling kernel matrix, Gup')
    Gup = matrix_assembler.assemble_matrix(k_t)
    # Interior problem so we subtract -(1/2)*M
    Gup -= 0.5 * mass_matrix.M

        # with open('Guu.matrix', 'wb') as f:
        #     cPickle.dump(Guu, f)
        # with open('Gup.matrix', 'wb') as f:
        #     cPickle.dump(Gup, f)



    def full_surface_traction(x, n):
        return np.array((x[0], x[1]))

    def section_traction(x):
        if np.abs(x[0]) < np.cos(24 * (np.pi / 50)):
            x_length = np.sqrt(x.dot(x))
            return -x / x_length
        return np.array((0.0, 0.0))

    # Gup multiplies displacements
    # Guu mutliplies tractions
    def solve_displacement_problem(fnc):
        tractions = tools.interpolate(lambda x,n: fnc(x), dh, bf, mesh)
        rhs = rhs_assembler.assemble_rhs(
                BasisFunctions.from_function(fnc), k_d)
        soln = np.linalg.solve(Gup, rhs)
        return soln, tractions

    disp, trac = solve_displacement_problem(section_traction)

    x, s = tools.evaluate_boundary_solution(400 / n_elements, disp,
                                                dh, bf, mesh)

    # Compute along the x axis
    x_vals = np.linspace(0, 1.0, 11)[:-1]

    # TODO: Spend some time refactoring and refining RHS and
    # Interior integrals. They don't require producing a matrix and thus are
    # simpler in a sense.
    interior_k_d = AdjointTractionVolumeKernel(1.0, 0.25)
    interior_k_t = HypersingularVolumeKernel(1.0, 0.25)

    ip = InteriorPoint(mesh,
                       bf, #BasisFunctions.from_function(section_traction),
                       k_d, k_t,
                       interior_k_d, interior_k_t, dh,
                       QuadGauss(interior_quad_pts, 0.0, 1.0))
    ones_like_disp = np.ones_like(disp)
    int_strs = np.array(
            [ip.compute_stress(
                (x_v, 0.0), disp, trac) for x_v in x_vals])
    int_disp = np.array(
            [ip.compute_displacement(
                (x_v, 0.0), disp, trac) for x_v in x_vals])

    sigma_xx_exact = np.array([0.0398, 0.0382, 0.0339, 0.0278, 0.0209,
                      0.0144, 0.0089, 0.0047, 0.0019, 0.0004])
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
        plt.plot(x_vals, int_strs[:, 0, 0], label = r'$\sigma_{xx}$')
        # plt.figure(5)
        plt.plot(x_vals, int_strs[:, 1, 1], label = r'$\sigma_{yy}$')
        plt.plot(x_vals, sigma_xx_exact, label = 'exact')
        plt.xlabel('distance along x axis from origin')
        plt.ylabel(r'$\sigma_{xx}$ and $\sigma_{yy}$')
        # plt.figure(6)
        # plt.plot(y_vals, int_disp[:, 0])
        # plt.figure(7)
        # plt.plot(y_vals, int_disp[:, 1])
        # plt.show()
    return int_strs[:, 0, 0]
    # See section 2.7 of starfield and crouch for the standard formulas to
    # convert from plane strain to plane stress.

if __name__ == "__main__":
    # 13 Quadrature points seems like enough for interior computations
    sigma_xx = main(50, 1, True, 12, 'Disc')
    sigma_xx_exact = np.array([0.0398, 0.0382, 0.0339, 0.0278, 0.0209,
                      0.0144, 0.0089, 0.0047, 0.0019, 0.0004])
    sigma_xx_crouch_100 = np.array([0.0393, 0.0378, 0.0335, 0.0274, 0.0206,
                       0.0141, 0.0086, 0.0044, 0.0016, 0.0000])
    sigma_xx_crouch_200 = np.array([0.0396, 0.0380, 0.0337, 0.0276, 0.0208,
                       0.0142, 0.0087, 0.0045, 0.0018, 0.0002])
    print tools.rmse(sigma_xx, sigma_xx_exact)
    print tools.rmse(sigma_xx_crouch_100, sigma_xx_exact)
    print tools.rmse(sigma_xx_crouch_200, sigma_xx_exact)
    # main(50, 1, True, 13, 'Disc')
    # main(50, 2, True, 13, 'Disc')
    # main(50, 4, True, 13, 'Disc')
    # main(100, 1, True, 13, 'Cont')
    plt.show()
    # strs = []
    # for i in range(5):
    #     strs.append(main(400, 1, False, i + 13))
    # import ipdb; ipdb.set_trace()

    # strs = []
    # for i in range(4):
    #     strs.append(main((i + 1) * 100, 1, False, 13))
    # import ipdb; ipdb.set_trace()

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
