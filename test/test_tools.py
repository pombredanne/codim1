import numpy as np
from codim1.core.tools import interpolate, plot_mesh, L2_error
from codim1.core import *
from codim1.post import evaluate_boundary_solution

def test_L2error():
    x1 = np.array([1.0, 2.0])
    x2 = np.array([0.5, 0.5])
    exact = np.sqrt(0.25 + 2.25) / (np.sqrt(0.5))
    np.testing.assert_almost_equal(exact, L2_error(x1, x2), 5)

def int_with_deg(n_elements, element_deg):
    msh = simple_line_mesh(n_elements)
    bf = basis_from_degree(element_deg)
    apply_to_elements(msh, "basis", bf, non_gen = True)
    apply_to_elements(msh, "continuous", False, non_gen = True)
    init_dofs(msh)
    fnc = lambda x, n: (x[0], x[1])
    val = interpolate(fnc, msh)
    return val

def test_interpolate():
    val = int_with_deg(2, 0)
    # The second half should be zero, because simple_line_mesh is
    # completely on the x axis.
    assert((val[2 * (1 + 1):] == 0).all())
    assert(val[0] == -0.5)
    assert(val[1] == 0.5)

def test_linear_interpolate():
    val = int_with_deg(2, 1)

    # The second half should be zero, because simple_line_mesh is
    # completely on the x axis.
    assert((val[2 * (1 + 1):] == 0).all())
    np.testing.assert_almost_equal(val[0:4], [-1.0, 0.0, 0.0, 1.0])

def test_interpolate_evaluate_hard():
    n_elements = 5
    # Sixth order elements should exactly interpolate a sixth order polynomial.
    element_deg = 6
    msh = simple_line_mesh(n_elements)
    bf = basis_from_degree(element_deg)
    apply_to_elements(msh, "basis", bf, non_gen = True)
    apply_to_elements(msh, "continuous", False, non_gen = True)
    init_dofs(msh)

    fnc = lambda x, n: (x[0] ** 6, 0)
    solution = interpolate(fnc, msh)
    x, soln = evaluate_boundary_solution(5, solution, msh)
    assert(x[-2][0] == 0.9)
    np.testing.assert_almost_equal(soln[-2][0], (0.9 ** 6))

def test_interpolate_normal():
    n_elements = 2
    element_deg = 0
    msh = simple_line_mesh(n_elements)
    bf = basis_from_degree(element_deg)
    apply_to_elements(msh, "basis", bf, non_gen = True)
    apply_to_elements(msh, "continuous", False, non_gen = True)
    init_dofs(msh)
    fnc = lambda x, n: (x[0] * n[0], x[1])
    val = interpolate(fnc, msh)

    # All zero!
    assert((val[:] == 0).all())


