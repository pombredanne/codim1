import numpy as np
import matplotlib.pyplot as plt
from codim1.core.dof_handler import DiscontinuousDOFHandler, ContinuousDOFHandler
from codim1.core.mesh import Mesh
from codim1.core.assembler import Assembler
from codim1.core.basis_funcs import BasisFunctions
from codim1.fast.elastic_kernel import ElastostaticKernel
from codim1.core.quad_strategy import QuadStrategy
import codim1.core.tools as tools

def main(n_elements, element_deg, plot):
    # Elastic parameters
    shear_modulus = 1.0
    poisson_ratio = 0.25

    # Quadrature points for the various circumstances
    quad_points_nonsingular = 7
    quad_points_logr = 8
    quad_points_oneoverr = 8


    #TODO: Implement isoparametric elements
    bf = BasisFunctions.from_degree(element_deg)
    # mesh = Mesh.simple_line_mesh(n_elements, -2.0, 2.0)

    # Is this an interior or exterior mesh?
    mesh = Mesh.circular_mesh(n_elements, 1.0)

    qs = QuadStrategy(mesh, quad_points_nonsingular, quad_points_logr,
                        quad_points_oneoverr)

    # tools.plot_mesh(mesh)
    kernel = ElastostaticKernel(shear_modulus, poisson_ratio)

    dh = ContinuousDOFHandler(mesh, element_deg)
    # dh = DiscontinuousDOFHandler(mesh, element_deg)
    assembler = Assembler(mesh, bf, kernel, dh, qs)

    H, G = assembler.assemble()

    # def fnc(x):
    #     if np.abs(x[0]) > 1.0:
    #         return (0.0, 0.0)
    #     else:
    #         return (0.0, np.exp(-(x[0]**2)*3))
    def fnc(x):
        # if np.abs(x[0]) < np.cos(24 * (np.pi / 50)):
        return (x[0], x[1])
        # return (0.0, 0.0)

    # Solve a dirichlet problem (specify displacement, derive traction)
    displacements = tools.interpolate(fnc, dh, bf, mesh)
    rhs = np.dot(H, displacements)
    soln = np.linalg.solve(G, rhs)
    x, s = tools.evaluate_boundary_solution(400 / n_elements, soln, dh, bf, mesh)

    traction_mag = np.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2)

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
        plt.plot(x[:, 0], traction_mag)

    return s
        # plt.show()
    # See section 2.7 of starfield and crouch for the standard formulas to convert
    # from plane strain to plane stress.

if __name__ == "__main__":
    import time
    start = time.time()
    s1 = main(10, 1, False)
    s2 = main(20, 1, False)
    s4 = main(40, 1, False)
    s8 = main(80, 1, True)
    import ipdb;ipdb.set_trace()

    end = time.time()
    print("Time: " + str(end - start))
    plt.show()
    # main(5)
