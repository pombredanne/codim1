import time
import sys

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
                        AdjointTractionKernel, HypersingularKernel, \
                        RegularizedHypersingularKernel
from codim1.core.quad_strategy import QuadStrategy
from codim1.core.quadrature import QuadGauss
from codim1.core.mass_matrix import MassMatrix
from codim1.core.interior_point import InteriorPoint
import codim1.core.tools as tools

# The theta width over which to apply the load to our cylinder.
alpha = (1 / 50.) * np.pi

def section_traction(x):
    # Only apply tractions over the arcs near y=1, y=-1
    if np.abs(x[0]) < np.cos((np.pi / 2) - alpha):
        x_length = np.sqrt(x.dot(x))
        return -x / x_length
    return np.array((0.0, 0.0))

# Found the exact solution in Frangi, Novati 1996 -- could copy that over to
# do better convergence tests, etc.

def disk(n_elements, element_deg, plot):
    # Elastic parameters
    shear_modulus = 1.0
    poisson_ratio = 0.25

    # Quadrature points for the various circumstances
    quad_min = 4
    quad_max = 10
    quad_logr = 10
    quad_oneoverr = 10

    # Define the solution basis functions
    bf = BasisFunctions.from_degree(element_deg)

    # We must use at least first order mesh basis functions.
    mesh_bf = BasisFunctions.from_degree(
              element_deg if element_deg > 0 else 1)

    # A circle with radius one.
    mesh = Mesh.circular_mesh(n_elements, 1.0, mesh_bf)
    # tools.plot_mesh(mesh)

    # This object defines what type of quadrature to use for different
    # situations (log(r) singular, 1/r singular, adjacent elements, others)
    # and how many points to use.
    qs = QuadStrategy(mesh, quad_max, quad_max, quad_logr, quad_oneoverr)

    # Maybe use more point for the RHS. This is because it is discontinuous
    # at: theta = (24/50) * pi, so if there aren't enough points, I will
    # miss that discontinuity
    # This is not necessary if the mesh is aligned with the discontinuity
    qs_rhs = qs#QuadStrategy(mesh, quad_max, quad_max,
                                 #quad_logr, quad_oneoverr)

    # The first two are elastic kernels for the displacement BIE,
    # these will be used to solve the BIE
    # The next two are elastic kernels for the traction BIE,
    # these will be used to compute interior stresses
    k_d = DisplacementKernel(shear_modulus, poisson_ratio)
    k_t = TractionKernel(shear_modulus, poisson_ratio)
    k_ta = AdjointTractionKernel(shear_modulus, poisson_ratio)
    k_rh = RegularizedHypersingularKernel(shear_modulus, poisson_ratio)
    k_h = HypersingularKernel(shear_modulus, poisson_ratio)

    # For a problem where the unknowns are displacements, use
    # a discontinuous basis. If the unknowns were tractions, a
    # continuous basis is used. These are constraints that follow from
    # some functional analysis fiddle-diddle
    dh = DiscontinuousDOFHandler(mesh, element_deg)

    # Assemble the matrix of displacements induced by displacements at
    # another location.
    # Gup multiplies displacements
    # Guu mutliplies tractions
    print('Assembling kernel matrix, Gup')
    matrix_assembler = MatrixAssembler(mesh, bf, dh, qs)
    Gup = matrix_assembler.assemble_matrix(k_t)

    # This mass matrix term arises from considering the cauchy singular form
    # of the Gup matrix.
    mass_matrix = MassMatrix(mesh, bf, bf,
                             dh, QuadGauss(element_deg + 1),
                             compute_on_init = True)
    Gup -= 0.5 * mass_matrix.M


    # Make the input function behave like a basis -- for internal reasons,
    # this makes assembly easier.
    # TODO: Could be moved inside assemble_rhs
    traction_function = BasisFunctions.from_function(section_traction)

    # Assemble the rhs, composed of the displacements induced by the
    # traction inputs.
    print("Assembling RHS")
    rhs_assembler = RHSAssembler(mesh, bf, dh, qs_rhs)
    rhs = rhs_assembler.assemble_rhs(traction_function, k_d)

    # Solve Ax = b, where x are the coefficients over the solution basis
    soln_coeffs = np.linalg.solve(Gup, rhs)

    # Create a solution object that pairs the coefficients with the basis
    soln = Solution(bf, dh, soln_coeffs)

    # Evaluate that solution at 400 points around the circle
    x, u = tools.evaluate_boundary_solution(400 / n_elements, soln, mesh)

    # Now, compute some interior values of stress along the x axis
    print("Performing interior computations.")
    x_vals = np.linspace(0, 1.0, 11)[:-1]

    # Use a slightly higher order quadrature for interior point
    # computations just to avoid problems in testing. Could be reduced
    # in the future.
    ip = InteriorPoint(mesh, dh, qs)

    # Get the tractions on the y-z plane (\sigma_xx, \sigma_xy)
    # where the normal is n_x=1, n_y=0
    normal = np.array([1.0, 0.0])
    # Positive contribution of the AdjointTraction kernel
    int_strs_x = np.array(
            [ip.compute((x_v, 0.0), normal, k_ta, traction_function)
             for x_v in x_vals])
    # Negative contribution of the hypersingular kernel
    int_strs_x2 = -np.array(
            [ip.compute((x_v, 0.0), normal, k_h, soln)
             for x_v in x_vals])
    # soln_deriv = Solution(bf.get_gradient_basis(mesh), dh, soln_coeffs)
    # int_strs_x2 = -np.array(
    #         [ip.compute((x_v, 0.0), normal, k_rh, soln_deriv)
    #          for x_v in x_vals])
    print int_strs_x2
    int_strs_x += int_strs_x2

    # Get the tractions on the x-z plane (\sigma_xy, \sigma_yy)
    normal = np.array([0.0, 1.0])
    # Negative contribution of the AdjointTraction kernel
    int_strs_y = np.array(
            [ip.compute((x_v, 0.0), normal, k_ta, traction_function)
             for x_v in x_vals])

    # Positive contribution of the hypersingular kernel
    int_strs_y -= np.array(
            [ip.compute((x_v, 0.0), normal, k_h, soln)
             for x_v in x_vals])

    sigma_xx_exact = np.array([0.0398, 0.0382, 0.0339, 0.0278, 0.0209,
                      0.0144, 0.0089, 0.0047, 0.0019, 0.0004])

    # Finally, plot some of the results.
    if plot:
        # u_x
        plt.figure(2)
        plt.plot(x[:, 0], u[:, 0])
        plt.xlabel(r'X')
        plt.ylabel(r'$u_x$', fontsize = 18)

        # u_y
        plt.figure(3)
        plt.plot(x[:, 0], u[:, 1])
        plt.xlabel(r'X')
        plt.ylabel(r'$u_y$', fontsize = 18)

        plt.figure(4)
        # \sigma_xx
        plt.plot(x_vals, int_strs_x[:, 0], linewidth = '2', label = r'$\sigma_{xx}$')
        # \sigma_yy
        plt.plot(x_vals, int_strs_y[:, 1], linewidth = '2', label = r'$\sigma_{yy}$')
        plt.plot(x_vals, sigma_xx_exact, linewidth = '2', label = 'exact')
        plt.xlabel('distance along x axis from origin')
        plt.ylabel(r'$\sigma_{xx}$ and $\sigma_{yy}$')
        plt.legend()

    # Collect the displacements at an array of interior points.
    save_interior = False
    if save_interior:
        x_pts = 20
        y_pts = 22
        x = np.linspace(-0.95, 0.95, x_pts)
        y = np.linspace(-0.95, 0.95, y_pts)
        int_ux = np.zeros((y_pts, x_pts))
        int_uy = np.zeros((y_pts, x_pts))
        for i in range(x_pts):
            for j in range(y_pts):
                print i, j
                x_val = x[i]
                y_val = y[j]
                if ((x_val ** 2) + (y_val ** 2)) > 0.99:
                    int_ux[j, i] = 0
                    int_uy[j, i] = 0
                    continue
                traction_effect = ip.compute((x_val, y_val), np.zeros(2),
                                             k_d, traction_function)
                displacement_effect = -ip.compute((x_val, y_val),
                                                  np.zeros(2),
                                                  k_t, soln)
                int_ux[j, i] = traction_effect[0] + displacement_effect[0]
                int_uy[j, i] = traction_effect[1] + displacement_effect[1]
        int_u = np.array([int_ux, int_uy])
        with open('int_u_disk.pkl', 'wb') as f:
            cPickle.dump(int_u, f)

    return int_strs_x[:, 0]

def reload_and_postprocess():
    # Creates a pretty picture =)
    f = open('data/disk_compression/int_u_disk.pkl', 'rb')
    int_u = cPickle.load(f)
    x_pts = 20
    y_pts = 22
    x = np.linspace(-0.95, 0.95, x_pts)
    y = np.linspace(-0.95, 0.95, y_pts)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    quiver_plot = plt.quiver(X, Y, int_u[0, :, :], int_u[1, :, :])
    plt.quiverkey(quiver_plot, 0.60, 0.95, 0.05, r'0.05', labelpos='W')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.ylim([-1.1, 1.2])
    plt.xlim([-1.1, 1.1])
    circ = plt.Circle((0, 0), radius = 1.0, color = 'g', fill = False)
    ax.add_patch(circ)
    plt.title(r'Displacement vectors for a disk compressed in plane strain')
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'reload':
        reload_and_postprocess()
        sys.exit()

    start = time.time()
    sigma_xx = disk(50, 0, True)
    end = time.time()
    print("Took: " + str(end - start) + " seconds")
    plt.show()

    # Calculate errors and compare with the crouch errors.
    sigma_xx_exact_perturbed = np.array([0.0398, 0.0382,
                                         0.0339, 0.0278, 0.0209,
                                         0.0144, 0.0089, 0.0047,
                                         0.0019, 0.0004]) + \
                               np.random.rand(10) * 0.00005
    sigma_xx_exact = np.array([0.0398, 0.0382, 0.0339, 0.0278, 0.0209,
                      0.0144, 0.0089, 0.0047, 0.0019, 0.0004])
    sigma_xx_crouch_100 = np.array([0.0393, 0.0378, 0.0335, 0.0274, 0.0206,
                       0.0141, 0.0086, 0.0044, 0.0016, 0.0000])
    sigma_xx_crouch_200 = np.array([0.0396, 0.0380, 0.0337, 0.0276, 0.0208,
                       0.0142, 0.0087, 0.0045, 0.0018, 0.0002])
    print tools.L2_error(sigma_xx, sigma_xx_exact)
    print tools.L2_error(sigma_xx_crouch_100, sigma_xx_exact)
    print tools.L2_error(sigma_xx_crouch_200, sigma_xx_exact)

    plt.show()

