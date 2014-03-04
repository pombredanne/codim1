import numpy as np
from dof_handler import DOFHandler
from assembler import Assembler
from basis_funcs import BasisFunctions
from quadrature import QuadGauss
from mesh import Mesh
from elastic_kernel import ElastostaticKernel

def main():
    # Number of elements
    n_elements = 10

    # Degree of the polynomial basis to use. For example, 1 is a linear basis
    element_deg = 0

    # Dimension of problem
    dim = 2

    # Elastic parameters
    shear_modulus = 1.0
    poisson_ratio = 0.25

    # Quadrature points for the various circumstances
    quad_points_nonsingular = 4
    quad_points_logr = 4
    quad_points_oneoverr = 4


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
    bc_list = [('traction', fnc) for i in range(n_elements)]
    tractions = np.empty(dh.total_dofs)
    for k in range(n_elements):
        for i in range(element_deg + 1):
            dof_x = dh.dof_map[0, k, i]
            dof_y = dh.dof_map[1, k, i]
            ref_pt = bf.nodes[i]
            node_pt = mesh.get_physical_points(k, ref_pt)
            t = fnc(node_pt)
            tractions[dof_x] = t[0]
            tractions[dof_y] = t[1]
    print tractions
    rhs = np.dot(H, tractions)
    soln = np.linalg.solve(G, rhs)
    print soln

    # See section 2.7 of starfield and crouch for the standard formulas to convert
    # from plane strain to plane stress.

if __name__ == "__main__":
    main()


def interpolate(self, element_id, fnc):
    """
    Interpolate a function of (x, y) onto
    """
    pass
