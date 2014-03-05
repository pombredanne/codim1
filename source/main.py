import numpy as np
import matplotlib.pyplot as plt
from dof_handler import DOFHandler
from basis_funcs import BasisFunctions
from mesh import Mesh
from assembler import Assembler
from elastic_kernel import ElastostaticKernel
import tools

def main():
    # Number of elements
    n_elements = 2

    # Degree of the polynomial basis to use. For example, 1 is a linear basis
    element_deg = 0

    # Dimension of problem
    dim = 2

    # Elastic parameters
    shear_modulus = 1.0
    poisson_ratio = 0.25

    # Quadrature points for the various circumstances
    quad_points_nonsingular = 11
    quad_points_logr = 11
    quad_points_oneoverr = 11


    bf = BasisFunctions.from_degree(element_deg)
    mesh = Mesh.simple_line_mesh(n_elements)
    kernel = ElastostaticKernel(shear_modulus, poisson_ratio)
    dh = DOFHandler(2, n_elements, element_deg)
    assembler = Assembler(mesh, bf, kernel, dh,
                          quad_points_nonsingular,
                          quad_points_logr,
                          quad_points_oneoverr)
    G, H = assembler.assemble()
    # tools.plot_matrix(G, 'G', show = False)
    # tools.plot_matrix(H, 'H')


    # Specify which boundary conditions we have for each element
    # Traction for every element with t_x = 0, t_y = 1.0
    fnc = lambda x: (0.0, 1.0)
    tractions = tools.interpolate(fnc, dh, bf, mesh)
    # bc_list = [('traction', fnc) for i in range(n_elements)]

    import ipdb; ipdb.set_trace()
    rhs = np.dot(H, tractions)
    soln = np.linalg.solve(G, rhs)

    x, u = tools.evaluate_boundary_solution(5, soln, dh, bf, mesh)

    plt.figure()
    plt.plot(x[:, 0], u[:, 0])
    plt.xlabel(r'X')
    plt.ylabel(r'$u_x$', fontsize = 18)
    plt.figure()
    plt.plot(x[:, 0], u[:, 1])
    plt.xlabel(r'X')
    plt.ylabel(r'$u_y$', fontsize = 18)
    plt.show()


    # See section 2.7 of starfield and crouch for the standard formulas to convert
    # from plane strain to plane stress.

if __name__ == "__main__":
    main()
