from codim1.core import *
from codim1.post import *
from codim1.core.tools import interpolate

import numpy as np

def test_evaluate_boundary_solution_easy():
    n_elements = 2
    element_deg = 0
    msh = simple_line_mesh(n_elements)
    bf = basis_from_degree(element_deg)
    apply_to_elements(msh, "basis", bf, non_gen = True)
    apply_to_elements(msh, "continuous", False, non_gen = True)
    init_dofs(msh)

    fnc = lambda x, n: (x[0], x[1])
    solution = interpolate(fnc, msh)
    apply_to_elements(msh, "bc", BC("traction", msh.elements[0].basis),
                      non_gen = True)
    x, u, t = evaluate_boundary_solution(msh, solution, 11)
    # Constant basis, so it should be 0.5 everywhere on the element [0,1]
    assert(x[0][-2] == 0.9)
    assert(u[0][-2] == 0.5)
    assert(u[1][-2] == 0.0)

def test_evaluate_solution_on_element():
    n_elements = 2
    element_deg = 1
    msh = simple_line_mesh(n_elements)
    bf = basis_from_degree(element_deg)
    apply_to_elements(msh, "basis", bf, non_gen = True)
    apply_to_elements(msh, "continuous", False, non_gen = True)
    init_dofs(msh)

    fnc = lambda x, n: (x[0], x[1])
    solution = interpolate(fnc, msh)
    msh.elements[1].bc = BC("traction", msh.elements[1].basis)
    u, t = evaluate_solution_on_element(msh.elements[1], 1.0, solution)
    assert(u[0] == 1.0)
    assert(u[1] == 0.0)

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
    apply_to_elements(msh, "bc", BC("traction", msh.elements[0].basis),
                      non_gen = True)
    x, u, t = evaluate_boundary_solution(msh, solution, 5)
    assert(x[0][-2] == 0.9)
    np.testing.assert_almost_equal(u[0][-2], (0.9 ** 6))

