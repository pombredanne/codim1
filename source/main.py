import numpy as np
import matplotlib.pyplot as plt
from dof_handler import DiscontinuousDOFHandler, ContinuousDOFHandler
from basis_funcs import BasisFunctions
from mesh import Mesh
from assembler import Assembler
from elastic_kernel import ElastostaticKernel
import tools

def main():
    # Should we plot
    plot = True

    # Number of elements
    n_elements = 30

    # Degree of the polynomial basis to use. For example, 1 is a linear basis
    element_deg = 1

    # Dimension of problem
    dim = 2

    # Elastic parameters
    shear_modulus = 1.0
    poisson_ratio = 0.25

    # Quadrature points for the various circumstances
    quad_points_nonsingular = 12
    quad_points_logr = 12
    quad_points_oneoverr = 12


    bf = BasisFunctions.from_degree(element_deg)
    mesh = Mesh.simple_line_mesh(n_elements, -2.0, 2.0)
    kernel = ElastostaticKernel(shear_modulus, poisson_ratio)
    dh = ContinuousDOFHandler(mesh, element_deg)
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


    def fnc(x):
        if np.abs(x[0]) > 1.0:
            return (0.0, 0.0)
        else:
            return (0.0, np.exp(-(x[0]**2)*3))
    # Solve a dirichlet problem (specify displacement, derive traction)
    tractions = tools.interpolate(fnc, dh, bf, mesh)
    rhs = np.dot(H, tractions)
    soln = np.linalg.solve(G, rhs)
    x, s = tools.evaluate_boundary_solution(20, soln, dh, bf, mesh)

    if plot:
        plt.figure()
        plt.plot(x[:, 0], s[:, 0])
        plt.xlabel(r'X')
        plt.ylabel(r'$t_x$', fontsize = 18)
        plt.figure()
        plt.plot(x[:, 0], s[:, 1])
        plt.xlabel(r'X')
        plt.ylabel(r'$t_y$', fontsize = 18)
        plt.show()

    import ipdb; ipdb.set_trace()
    # See section 2.7 of starfield and crouch for the standard formulas to convert
    # from plane strain to plane stress.

if __name__ == "__main__":
    main()
