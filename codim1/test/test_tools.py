import numpy as np
from codim1.core.tools import interpolate, evaluate_boundary_solution, \
                evaluate_solution_on_element, plot_mesh, L2_error
from codim1.core.dof_handler import DiscontinuousDOFHandler
from codim1.core.basis_funcs import BasisFunctions, Solution
from codim1.core.mesh import Mesh

# def test_plot_mesh():
#     m = Mesh.circular_mesh(50, 1.0)
#     plot_mesh(m)

def test_L2error():
    x1 = np.array([1.0, 2.0])
    x2 = np.array([0.5, 0.5])
    exact = np.sqrt(0.25 + 2.25) / (np.sqrt(0.5))
    np.testing.assert_almost_equal(exact, L2_error(x1, x2), 5)

def test_interpolate():
    n_elements = 2
    element_deg = 0
    msh = Mesh.simple_line_mesh(n_elements)
    bf = BasisFunctions.from_degree(element_deg)
    dh = DiscontinuousDOFHandler(msh, bf)
    fnc = lambda x, n: (x[0], x[1])
    val = interpolate(fnc, dh, bf, msh)

    # The second half should be zero, because simple_line_mesh is
    # completely on the x axis.
    assert((val[n_elements * (element_deg + 1):] == 0).all())
    assert(val[0] == -0.5)
    assert(val[1] == 0.5)

def test_evaluate_boundary_solution_easy():
    n_elements = 2
    element_deg = 0
    msh = Mesh.simple_line_mesh(n_elements)
    bf = BasisFunctions.from_degree(element_deg)
    dh = DiscontinuousDOFHandler(msh, bf)
    fnc = lambda x, n: (x[0], x[1])
    solution_coeffs = interpolate(fnc, dh, bf, msh)
    solution = Solution(bf, dh, solution_coeffs)
    x, soln = evaluate_boundary_solution(11, solution, msh)
    # Constant basis, so it should be 0.5 everywhere on the element [0,1]
    assert(x[-2][0] == 0.9)
    assert(soln[-2][0] == 0.5)
    assert(soln[-2][1] == 0.0)

def test_evaluate_solution_on_element():
    n_elements = 2
    element_deg = 1
    msh = Mesh.simple_line_mesh(n_elements)
    bf = BasisFunctions.from_degree(element_deg)
    dh = DiscontinuousDOFHandler(msh, bf)
    fnc = lambda x, n: (x[0], x[1])
    solution_coeffs = interpolate(fnc, dh, bf, msh)
    solution = Solution(bf, dh, solution_coeffs)
    eval = evaluate_solution_on_element(1, 1.0, solution, msh)
    assert(eval[0] == 1.0)
    assert(eval[1] == 0.0)


def test_interpolate_evaluate_hard():
    n_elements = 5
    # Sixth order elements should exactly interpolate a sixth order polynomial.
    element_deg = 6
    msh = Mesh.simple_line_mesh(n_elements)
    bf = BasisFunctions.from_degree(element_deg)
    dh = DiscontinuousDOFHandler(msh, bf)
    fnc = lambda x, n: (x[0] ** 6, 0)
    solution_coeffs = interpolate(fnc, dh, bf, msh)
    solution = Solution(bf, dh, solution_coeffs)
    x, soln = evaluate_boundary_solution(5, solution, msh)
    assert(x[-2][0] == 0.9)
    np.testing.assert_almost_equal(soln[-2][0], (0.9 ** 6))

def test_interpolate_normal():
    n_elements = 2
    element_deg = 0
    msh = Mesh.simple_line_mesh(n_elements)
    bf = BasisFunctions.from_degree(element_deg)
    dh = DiscontinuousDOFHandler(msh, bf)
    fnc = lambda x, n: (x[0] * n[0], x[1])
    val = interpolate(fnc, dh, bf, msh)

    # All zero!
    assert((val[:] == 0).all())


