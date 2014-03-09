import numpy as np
import matplotlib.pyplot as plt

# def plot_kernel(kernel_fnc,
#     x = np.linspace(0.0, 1.0, 100)
#     y = np.linspace(0.0, 1.0, 100)
#     X, Y = np.meshgrid(x, y)
#     a = np.zeros_like(X)
#     b = np.zeros_like(X)
#     for i in range(X.shape[0]):
#         for j in range(X.shape[1]):
#             a[i, j] = f(x[i], y[j])[1, 1]
#             b[i, j] = ff(x[i], y[j])
#     import matplotlib.pyplot as plt
#     plt.figure()
#     plt.imshow(a)
#     plt.colorbar()
#     plt.figure()
#     plt.imshow(b)
#     plt.colorbar()
#     plt.show()

def plot_mesh(msh, show = True):
    points = msh.vertices[msh.element_to_vertex[:, 0]]
    import ipdb;ipdb.set_trace()
    x = points[:, 0]
    y = points[:, 1]
    plt.plot(x, y)
    if show:
        plt.show()

def plot_matrix(M, title, show = True):
    """
    Creates and shows two plots. One of the matrix itself and
    one of the symmetric pattern M - M.T
    """
    plt.figure()
    plt.imshow(M)
    plt.title(title)
    plt.colorbar()

    plt.figure()
    plt.imshow((M - M.T) / (M + M.T))
    plt.title(title)
    plt.colorbar()

    if show:
        plt.show()


def interpolate(fnc, dof_handler, basis_funcs, mesh):
    """
    Interpolate the value of fnc onto the polynomial basis function space
    defined by basis_funcs and mesh. This simply sets the coefficient of each
    basis function to be the value of fnc at the node (where phi(x) = 1).
    """
    result = np.empty(dof_handler.total_dofs)
    for k in range(mesh.n_elements):
        for i in range(basis_funcs.num_fncs):
            dof_x = dof_handler.dof_map[0, k, i]
            dof_y = dof_handler.dof_map[1, k, i]
            ref_pt = basis_funcs.nodes[i]
            node_pt = mesh.get_physical_points(k, ref_pt)
            f_val = fnc(node_pt)
            result[dof_x] = f_val[0]
            result[dof_y] = f_val[1]
    return result

def evaluate_boundary_solution(points_per_element, soln,
                               dof_handler, basis_funcs, mesh):
    """
    Once a solution is computed, it's often nice to know the actual value, not
    just the coefficients of the polynomial basis! This function will produce
    1 point for every "point_separation" of distance along the boundary.

    This accepts a vector-valued solution and produces vector-valued point
    evaluations. A scalar version would be simple to write with this as a
    template.
    """
    x = []
    y = []
    for k in range(mesh.n_elements):
        for pt in np.linspace(0.0, 1.0, points_per_element):
            x.append(mesh.get_physical_points(k, pt))
            ux = 0
            uy = 0
            for i in range(dof_handler.element_deg + 1):
                coeff = dof_handler.dof_map[:, k, i]
                ux += soln[coeff[0]] * basis_funcs.evaluate_basis(i, pt)
                uy += soln[coeff[1]] * basis_funcs.evaluate_basis(i, pt)
            y.append([ux, uy])
    x = np.array(x)
    y = np.array(y)
    return x, y

def evaluate_solution_on_element(element_idx, reference_point, soln,
                                 dof_handler, basis_funcs, mesh):
    soln_x = 0.0
    soln_y = 0.0
    for i in range(basis_funcs.num_fncs):
        dof_x = dof_handler.dof_map[0, element_idx, i]
        dof_y = dof_handler.dof_map[1, element_idx, i]
        soln_x += soln[dof_x] * basis_funcs.evaluate_basis(i, reference_point)
        soln_y += soln[dof_y] * basis_funcs.evaluate_basis(i, reference_point)
    return np.array([soln_x, soln_y])

################################################################################
# TESTS                                                                        #
################################################################################
from dof_handler import DiscontinuousDOFHandler
from basis_funcs import BasisFunctions
from mesh import Mesh

# def test_plot_mesh():
#     m = Mesh.circular_mesh(50, 1.0)
#     plot_mesh(m)

def test_interpolate():
    n_elements = 2
    element_deg = 0
    bf = BasisFunctions.from_degree(element_deg)
    msh = Mesh.simple_line_mesh(n_elements)
    dh = DiscontinuousDOFHandler(msh, element_deg)
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
    dh = DiscontinuousDOFHandler(msh, element_deg)
    fnc = lambda x: (x[0], x[1])
    solution = interpolate(fnc, dh, bf, msh)
    x, soln = evaluate_boundary_solution(11, solution, dh, bf, msh)
    # Constant basis, so it should be 0.5 everywhere on the element [0,1]
    assert(x[-2][0] == 0.9)
    assert(soln[-2][0] == 0.5)
    assert(soln[-2][1] == 0.0)

def test_evaluate_solution_on_element():
    n_elements = 2
    element_deg = 1
    bf = BasisFunctions.from_degree(element_deg)
    msh = Mesh.simple_line_mesh(n_elements)
    dh = DiscontinuousDOFHandler(msh, element_deg)
    fnc = lambda x: (x[0], x[1])
    solution = interpolate(fnc, dh, bf, msh)
    eval = evaluate_solution_on_element(1, 1.0, solution,
                                 dh, bf, msh)
    assert(eval[0] == 1.0)
    assert(eval[1] == 0.0)


def test_interpolate_evaluate_hard():
    n_elements = 5
    # Sixth order elements should exactly interpolate a sixth order polynomial.
    element_deg = 6
    bf = BasisFunctions.from_degree(element_deg)
    msh = Mesh.simple_line_mesh(n_elements)
    dh = DiscontinuousDOFHandler(msh, element_deg)
    fnc = lambda x: (x[0] ** 6, 0)
    solution = interpolate(fnc, dh, bf, msh)
    x, soln = evaluate_boundary_solution(5, solution, dh, bf, msh)
    assert(x[-2][0] == 0.9)
    np.testing.assert_almost_equal(soln[-2][0], (0.9 ** 6))


