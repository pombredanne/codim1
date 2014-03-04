import numpy as np
import matplotlib.pyplot as plt
from dof_handler import DOFHandler
from basis_funcs import BasisFunctions
from mesh import Mesh
from assembler import Assembler
from elastic_kernel import ElastostaticKernel
from tools import interpolate

def main():
    # Number of elements
    n_elements = 20

    # Degree of the polynomial basis to use. For example, 1 is a linear basis
    element_deg = 1

    # Dimension of problem
    dim = 2

    # Elastic parameters
    shear_modulus = 1.0
    poisson_ratio = 0.25

    # Quadrature points for the various circumstances
    quad_points_nonsingular = 10
    quad_points_logr = 11
    quad_points_oneoverr = 10


    bf = BasisFunctions.from_degree(element_deg)
    mesh = Mesh.simple_line_mesh(n_elements)
    kernel = ElastostaticKernel(shear_modulus, poisson_ratio)
    dh = DOFHandler(2, n_elements, element_deg)
    assembler = Assembler(mesh, bf, kernel, dh,
                          quad_points_nonsingular,
                          quad_points_logr,
                          quad_points_oneoverr)
    G, H = assembler.assemble()
    # print G
    # print H

    # Specify which boundary conditions we have for each element
    # Traction for every element with t_x = 0, t_y = 1.0
    fnc = lambda x: (0.0, 1.0)
    tractions = interpolate(fnc, dh, bf, mesh)
    # bc_list = [('traction', fnc) for i in range(n_elements)]

    # print tractions

    rhs = np.dot(H, tractions)
    soln = np.linalg.solve(G, rhs)

    # print soln

    pts_per_element = 10
    pt_index = 0
    total_pts = pts_per_element * n_elements
    center_pt = np.zeros(total_pts)
    ux = np.zeros(total_pts)
    uy = np.zeros(total_pts)
    for k in range(n_elements):
        for pt in np.linspace(0.0, 1.0, pts_per_element):
            center_pt[pt_index] = mesh.get_physical_points(k, pt)[0][0]
            for i in range(element_deg + 1):
                coeff = dh.dof_map[:, k, i]
                ux[pt_index] += soln[coeff[0]] * bf.evaluate_basis(i, pt)
                uy[pt_index] += soln[coeff[1]] * bf.evaluate_basis(i, pt)
            pt_index += 1

    plt.figure()
    plt.plot(center_pt, ux)
    plt.figure()
    plt.plot(center_pt, uy)
    plt.show()


    # See section 2.7 of starfield and crouch for the standard formulas to convert
    # from plane strain to plane stress.

if __name__ == "__main__":
    main()


def interpolate(self, element_id, fnc):
    """
    Interpolate a function of (x, y) onto
    """
    pass
