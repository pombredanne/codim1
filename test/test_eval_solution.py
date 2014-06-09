from codim1.core import *
from codim1.post

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
    x, soln = evaluate_boundary_solution(11, solution, msh)
    # Constant basis, so it should be 0.5 everywhere on the element [0,1]
    assert(x[-2][0] == 0.9)
    assert(soln[-2][0] == 0.5)
    assert(soln[-2][1] == 0.0)

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
    eval = evaluate_solution_on_element(msh.elements[1], 1.0, solution)
    assert(eval[0] == 1.0)
    assert(eval[1] == 0.0)
