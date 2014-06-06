import sys
import numpy as np
import matplotlib.pyplot as plt

import codim1.core.tools as tools
from codim1.core import *
from codim1.assembly import *
from codim1.fast_lib import *

# The theta width over which to apply the load to our cylinder.
alpha = (1 / 50.) * np.pi
alpha = np.cos((np.pi / 2) - alpha)

def section_traction(x, n):
    # Only apply tractions over the arcs near y=1, y=-1
    if np.abs(x[0]) < alpha:
        x_length = np.sqrt(x[0] ** 2 + x[1] ** 2)
        return (-x[0] / x_length, -x[1] / x_length)
    return (0.0, 0.0)

# Less accurate, because it doesn't scale the return value, but it's faster
def less_accurate_section_traction(x, d):
    # Only apply tractions over the arcs near y=1, y=-1
    if abs(x[0]) < alpha:
        return -x[d]
    return 0.0

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
    bf = basis_from_degree(element_deg)

    # A circle with radius one.
    mesh = circular_mesh(n_elements, 1.0)
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

    # If I am using the displacement BIE, I use a discontinuous basis
    # If I am using the traction BIE (hypersingular), I use a continuous
    # basis

    apply_to_elements(mesh, "basis", bf, non_gen = True)
    apply_to_elements(mesh, "continuous", False, non_gen = True)
    apply_to_elements(mesh, "qs", qs, non_gen = True)
    init_dofs(mesh)

    # Assemble the matrix of displacements induced by displacements at
    # another location.
    # Gup multiplies displacements
    # Guu mutliplies tractions
    print('Assembling kernel matrix, Gup')
    Gup = simple_matrix_assemble(mesh, k_t)

    # This mass matrix term arises from considering the cauchy singular form
    # of the Gup matrix.
    Gup -= 0.5 * assemble_mass_matrix(mesh, gauss(element_deg + 1))


    traction_function = tools.interpolate(section_traction, mesh)
    apply_coeffs(mesh, traction_function, "trac_fnc")

    # Assemble the rhs, composed of the displacements induced by the
    # traction inputs.
    print("Assembling RHS")
    rhs = simple_rhs_assemble(mesh, lambda e: e.trac_fnc, k_d)

    # Solve Ax = b, where x are the coefficients over the solution basis
    soln_coeffs = np.linalg.solve(Gup, rhs)

    # Create a solution object that pairs the coefficients with the basis
    print("Solving System")
    apply_coeffs(mesh, soln_coeffs, "soln")

    # Evaluate that solution at 400 points around the circle
    x, u = tools.evaluate_boundary_solution(400 / n_elements,
                                            soln_coeffs, mesh)

    # Now, compute some interior values of stress along the x axis
    print("Performing interior computations.")
    x_vals = np.linspace(0, 1.0, 11)[:-1]

    # Get the tractions on the y-z plane (\sigma_xx, \sigma_xy)
    # where the normal is n_x=1, n_y=0
    normal = np.array([1.0, 0.0])
    # Positive contribution of the AdjointTraction kernel
    int_strs_x = -np.array(
            [interior_pt(mesh, [(x_v, 0.0), normal], k_ta,
                         basis_grabber = lambda e: e.trac_fnc)
             for x_v in x_vals])
    # Negative contribution of the hypersingular kernel
    int_strs_x -= np.array(
            [interior_pt(mesh, [(x_v, 0.0), normal], k_h,
                         basis_grabber = lambda e: e.soln)
             for x_v in x_vals])

    # Get the tractions on the x-z plane (\sigma_xy, \sigma_yy)
    normal = np.array([0.0, 1.0])
    # Negative contribution of the AdjointTraction kernel
    int_strs_y = -np.array(
            [interior_pt(mesh, [(x_v, 0.0), normal], k_ta,
                         basis_grabber = lambda e: e.trac_fnc)
             for x_v in x_vals])

    # Positive contribution of the hypersingular kernel
    int_strs_y -= np.array(
            [interior_pt(mesh, [(x_v, 0.0), normal], k_h,
                         basis_grabber = lambda e: e.soln)
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
#
#         plt.figure(4)
#         # \sigma_xx
#         plt.plot(x_vals, int_strs_x[:, 0], linewidth = '2', label = r'$\sigma_{xx}$')
#         # \sigma_yy
#         plt.plot(x_vals, int_strs_y[:, 1], linewidth = '2', label = r'$\sigma_{yy}$')
#         plt.plot(x_vals, sigma_xx_exact, linewidth = '2', label = 'exact')
#         plt.xlabel('distance along x axis from origin')
#         plt.ylabel(r'$\sigma_{xx}$ and $\sigma_{yy}$')
#         plt.legend()

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
                # print i, j
                x_val = x[i]
                y_val = y[j]
                if ((x_val ** 2) + (y_val ** 2)) > 0.99:
                    int_ux[j, i] = 0
                    int_uy[j, i] = 0
                    continue
                traction_effect = interior_pt(mesh,
                                [(x_val, y_val), np.zeros(2)], k_d,
                                basis_grabber = lambda e: e.trac_fnc)
                displacement_effect = -interior_pt(mesh,
                                [(x_val, y_val), np.zeros(2)], k_t,
                                basis_grabber = lambda e: e.soln)

                int_ux[j, i] = traction_effect[0] - displacement_effect[0]
                int_uy[j, i] = traction_effect[1] - displacement_effect[1]
        int_u = np.array([int_ux, int_uy])
        # with open('int_u_disk.pkl', 'wb') as f:
        #     cPickle.dump(int_u, f)
        reload_and_postprocess(int_u)

    return int_strs_x[:, 0]

def reload_and_postprocess(int_u = None):
    # Creates a pretty picture =)
    if int_u is None:
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


def test_disk_compression():
    if len(sys.argv) > 1 and sys.argv[1] == 'reload':
        reload_and_postprocess()
        sys.exit()

    sigma_xx = disk(50, 1, True)
    # print("Took: " + str(end - start) + " seconds")
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
    my_error =  tools.L2_error(sigma_xx, sigma_xx_exact)
    print my_error
    print tools.L2_error(sigma_xx_crouch_100, sigma_xx_exact)
    their_error = tools.L2_error(sigma_xx_crouch_200, sigma_xx_exact)
    print their_error
    assert(my_error < their_error)

    plt.show()

if __name__ == "__main__":
    test_disk_compression()

