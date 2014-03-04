import numpy as np

def interpolate(fnc, dof_handler, basis_funcs, mesh):
    """
    Interpolate the value of fnc onto the polynomial basis function space
    defined by basis_funcs and mesh. This simply sets the coefficient of each
    basis function to be the value of fnc at the node (where phi(x) = 1).
    """
    result = np.empty(dof_handler.total_dofs)
    for k in range(dof_handler.n_elements):
        for i in range(basis_funcs.num_fncs):
            dof_x = dof_handler.dof_map[0, k, i]
            dof_y = dof_handler.dof_map[1, k, i]
            ref_pt = basis_funcs.nodes[i]
            node_pt = mesh.get_physical_points(k, ref_pt)[0]
            f_val = fnc(node_pt)
            result[dof_x] = f_val[0]
            result[dof_y] = f_val[1]
    return result

def evaluate_boundary_solution(points, soln, mesh, basis_funcs, dof_handler):
    """
    Once a solution is computed, it's often nice to know the actual value, not
    just the coefficients of the polynomial basis! The function will return
    those values at the locations in "points".

    This accepts a vector-valued solution and produces vector-valued point
    evaluations. A scalar version would be simple to write with this as a
    template.
    """
    result_x = np.zeros_like(points, dim)
    for pt in points:

################################################################################
# TESTS                                                                        #
################################################################################
from dof_handler import DOFHandler
from basis_funcs import BasisFunctions
from mesh import Mesh

def test_interpolate():
    n_elements = 2
    element_deg = 0
    bf = BasisFunctions.from_degree(element_deg)
    msh = Mesh.simple_line_mesh(n_elements)
    dh = DOFHandler(2, n_elements, element_deg)
    fnc = lambda x: (x[0], x[1])
    val = interpolate(fnc, dh, bf, msh)

    # The second half should be zero, because simple_line_mesh is
    # completely on the x axis.
    assert((val[n_elements * (element_deg + 1):] == 0).all())
    assert(val[0] == -0.5)
    assert(val[1] == 0.5)

def test_evaluate_boundary_solution_easy():
    n_elements = 2
    element_deg = 0
    bf = BasisFunctions.from_degree(element_deg)
    msh = Mesh.simple_line_mesh(n_elements)
    dh = DOFHandler(2, n_elements, element_deg)
    fnc = lambda x: (x[0] * x[0], 0)
    solution = interpolate(fnc, dh, bf, msh)

def test_evaluate_boundary_solution_hard():
    n_elements = 5
    # Super high element degree should mean that the recovered solution
    # is, to high precision, equal to the original function value at a point
    element_deg = 10
    bf = BasisFunctions.from_degree(element_deg)
    msh = Mesh.simple_line_mesh(n_elements)
    dh = DOFHandler(2, n_elements, element_deg)
    fnc = lambda x: (x[0] * x[0], 0)
    solution = interpolate(fnc, dh, bf, msh)


