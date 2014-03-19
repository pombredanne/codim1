import time

import cPickle
import numpy as np
import matplotlib.pyplot as plt
from codim1.core.dof_handler import DiscontinuousDOFHandler,\
                                    ContinuousDOFHandler
from codim1.core.mesh import Mesh
from codim1.core.matrix_assembler import MatrixAssembler
from codim1.core.rhs_assembler import RHSAssembler
from codim1.core.basis_funcs import BasisFunctions, Solution
from codim1.fast.elastic_kernel import DisplacementKernel, TractionKernel, \
        AdjointTractionKernel, HypersingularKernel
from codim1.core.quad_strategy import QuadStrategy
from codim1.core.quadrature import QuadGauss
from codim1.core.mass_matrix import MassMatrix
# from codim1.core.interior_point import InteriorPoint
import codim1.core.tools as tools

def full_surface_traction(x, n):
    return np.array((x[0], x[1]))

def section_traction(x):
    if np.abs(x[0]) < np.cos(24 * (np.pi / 50)):
        x_length = np.sqrt(x.dot(x))
        return -x / x_length
    return np.array((0.0, 0.0))

def main(n_elements, element_deg, plot):
    # Elastic parameters
    shear_modulus = 1.0
    poisson_ratio = 0.25

    # Quadrature points for the various circumstances
    quad_min = 4
    quad_max = 10
    quad_logr = 10
    quad_oneoverr = 10
    # I did some experiments and
    # 13 Quadrature points seems like it gives error like 1e-10, lower
    # than necessary, but nice for testing other components
    interior_quad_pts = 13

    # Is this an interior or exterior mesh?
    mesh = Mesh.circular_mesh(n_elements, 1.0)

    #TODO: Implement isoparametric elements
    # Define the basis functions on the mesh.
    bf = BasisFunctions.from_degree(element_deg, mesh)

    # This object defines what type of quadrature to use for different
    # situations (log(r) singular, 1/r singular, adjacent elements, others)
    # and how many points to use.
    qs = QuadStrategy(mesh, quad_min, quad_max, quad_logr, quad_oneoverr)

    # The first two are elastic kernels for the displacement BIE,
    # these will be used to solve the BIE
    # The next two are elastic kernels for the traction BIE,
    # these will be used to compute interior stresses
    k_d = DisplacementKernel(shear_modulus, poisson_ratio)
    k_t = TractionKernel(shear_modulus, poisson_ratio)
    k_ta = AdjointTractionKernel(1.0, 0.25)
    k_h = HypersingularKernel(1.0, 0.25)

    # For a problem where the unknowns are displacements, we use
    # a discontinuous basis.
    dh = DiscontinuousDOFHandler(mesh, element_deg)


    # Assemble the matrix of displacements induced by displacements at
    # another location.
    # FYI
    # Gup multiplies displacements
    # Guu mutliplies tractions
    print('Assembling kernel matrix, Gup')
    matrix_assembler = MatrixAssembler(mesh, bf, dh, qs)
    Gup = matrix_assembler.assemble_matrix(k_t)

    # This mass matrix term arises from considering the cauchy singular form
    # of the Gup matrix.
    mass_matrix = MassMatrix(mesh, bf, dh, QuadGauss(element_deg + 1),
                             compute_on_init = True)
    Gup -= 0.5 * mass_matrix.M


    # Make the input function behave like a basis -- for internal reasons,
    # this makes assembly easier.
    # TODO: Could be moved inside assemble_rhs
    fnc = BasisFunctions.from_function(section_traction)
    # Assemble the rhs, composed of the displacements induced by the traction
    # inputs.
    rhs_assembler = RHSAssembler(mesh, bf, dh, qs)
    rhs = rhs_assembler.assemble_rhs(fnc, k_d)

    # Solve Ax = b, where x are the coefficients over the solution basis
    soln_coeffs = np.linalg.solve(Gup, rhs)

    # Create a solution object that pairs the coefficients with the basis
    soln = Solution(bf, dh, soln_coeffs)

    # Evaluate that solution at 400 points around the circle
    x, u = tools.evaluate_boundary_solution(400 / n_elements, soln, bf, mesh)

    # Now, compute some interior values of stress along the x axis
    x_vals = np.linspace(0, 1.0, 11)[:-1]


    # ip = InteriorPoint(mesh,
    #                    bf, #BasisFunctions.from_function(section_traction),
    #                    k_d, k_t,
    #                    interior_k_d, interior_k_t, dh,
    #                    QuadGauss(interior_quad_pts, 0.0, 1.0))
    # ones_like_disp = np.ones_like(disp)
    # int_strs = np.array(
    #         [ip.compute_stress(
    #             (x_v, 0.0), disp, trac) for x_v in x_vals])
    # int_disp = np.array(
    #         [ip.compute_displacement(
    #             (x_v, 0.0), disp, trac) for x_v in x_vals])

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
        # plt.figure(4)
        # plt.plot(x_vals, int_strs[:, 0, 0], label = r'$\sigma_{xx}$')
        # # plt.figure(5)
        # plt.plot(x_vals, int_strs[:, 1, 1], label = r'$\sigma_{yy}$')
        # plt.plot(x_vals, sigma_xx_exact, label = 'exact')
        # plt.xlabel('distance along x axis from origin')
        # plt.ylabel(r'$\sigma_{xx}$ and $\sigma_{yy}$')
        # plt.figure(6)
        # plt.plot(y_vals, int_disp[:, 0])
        # plt.figure(7)
        # plt.plot(y_vals, int_disp[:, 1])
        # plt.show()
    # return int_strs[:, 0, 0]
    # See section 2.7 of starfield and crouch for the standard formulas to
    # convert from plane strain to plane stress.

if __name__ == "__main__":
    sigma_xx = main(50, 1, True)
    plt.show()
    sigma_xx_exact = np.array([0.0398, 0.0382, 0.0339, 0.0278, 0.0209,
                      0.0144, 0.0089, 0.0047, 0.0019, 0.0004])
    sigma_xx_crouch_100 = np.array([0.0393, 0.0378, 0.0335, 0.0274, 0.0206,
                       0.0141, 0.0086, 0.0044, 0.0016, 0.0000])
    sigma_xx_crouch_200 = np.array([0.0396, 0.0380, 0.0337, 0.0276, 0.0208,
                       0.0142, 0.0087, 0.0045, 0.0018, 0.0002])
    print tools.rmse(sigma_xx, sigma_xx_exact)
    print tools.rmse(sigma_xx_crouch_100, sigma_xx_exact)
    print tools.rmse(sigma_xx_crouch_200, sigma_xx_exact)
    plt.show()

