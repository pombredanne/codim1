import numpy as np
import matplotlib.pyplot as plt
from dof_handler import DiscontinuousDOFHandler, ContinuousDOFHandler
from basis_funcs import BasisFunctions
from mesh import Mesh
from assembler import Assembler
from fast.elastic_kernel import ElastostaticKernel
import tools

def main(deg, plot):
    # Number of elements
    n_elements = 20

    # Degree of the polynomial basis to use. For example, 1 is a linear basis
    element_deg = deg

    # Dimension of problem
    dim = 2

    # Elastic parameters
    shear_modulus = 1.0
    poisson_ratio = 0.25

    # Quadrature points for the various circumstances
    quad_points_nonsingular = 8
    quad_points_logr = 8
    quad_points_oneoverr = 8


    #TODO: Implement isoparametric elements
    bf = BasisFunctions.from_degree(element_deg)
    # mesh = Mesh.simple_line_mesh(n_elements, -2.0, 2.0)

    # Is this an interior or exterior mesh?
    mesh = Mesh.circular_mesh(n_elements, 1.0)

    # tools.plot_mesh(mesh)
    kernel = ElastostaticKernel(shear_modulus, poisson_ratio)

    dh = ContinuousDOFHandler(mesh, element_deg)
    # dh = DiscontinuousDOFHandler(mesh, element_deg)
    assembler = Assembler(mesh, bf, kernel, dh,
                          quad_points_nonsingular,
                          quad_points_logr,
                          quad_points_oneoverr)
    H, G = assembler.assemble()
    # Setting the small values to zero just makes debugging easier
    H[np.abs(H) < 1e-14] = 0.0
    G[np.abs(G) < 1e-14] = 0.0
    # tools.plot_matrix(H, 'H', show = False)
    # tools.plot_matrix(G, 'G')


    # def fnc(x):
    #     if np.abs(x[0]) > 1.0:
    #         return (0.0, 0.0)
    #     else:
    #         return (0.0, np.exp(-(x[0]**2)*3))
    def fnc(x):
        return (x[0], x[1])
    # Solve a dirichlet problem (specify displacement, derive traction)
    displacements = tools.interpolate(fnc, dh, bf, mesh)
    rhs = np.dot(H, displacements)
    soln = np.linalg.solve(G, rhs)
    x, s = tools.evaluate_boundary_solution(20, soln, dh, bf, mesh)

    if plot:
        plt.figure(2)
        plt.plot(x[:, 0], s[:, 0])
        plt.xlabel(r'X')
        plt.ylabel(r'$t_x$', fontsize = 18)
        plt.figure(3)
        plt.plot(x[:, 0], s[:, 1])
        plt.xlabel(r'X')
        plt.ylabel(r'$t_y$', fontsize = 18)
        plt.figure(4)
        plt.plot(np.arctan(x[:(len(x) / 2), 1] / x[:(len(x) / 2), 0]),
                np.sqrt(s[:(len(x) / 2), 0] ** 2 + s[:(len(x) / 2), 1] ** 2))
        # plt.show()
    # See section 2.7 of starfield and crouch for the standard formulas to convert
    # from plane strain to plane stress.

if __name__ == "__main__":
    main(1, False)
    # main(5)
    # plt.show()
