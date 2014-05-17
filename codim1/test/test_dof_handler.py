import numpy as np
from codim1.core import *

def test_mixed_dof_handler():
    msh = simple_line_mesh(3)
    bf = BasisFunctions.from_degree(2)
    apply_to_elements(msh, "basis", bf, non_gen = True)
    apply_to_elements(msh, "continuous", True, non_gen = True)
    msh.elements[2].continuous = False
    init_dofs(msh)
    assert(msh.elements[0].dofs[0, 0] == 0)
    assert(msh.elements[1].dofs[0, 0] == 2)
    assert(msh.elements[2].dofs[0, 0] == 5)
    assert(msh.elements[2].dofs[0, 1] == 6)


def test_discontinuous_dof_handler():
    msh = simple_line_mesh(2)
    bf = BasisFunctions.from_degree(2)
    apply_to_elements(msh, "basis", bf, non_gen = True)
    apply_to_elements(msh, "continuous", False, non_gen = True)
    init_dofs(msh)
    assert(msh.elements[0].dofs[0, 0] == 0)
    assert(msh.elements[1].dofs[0, 1] == 4)
    assert(msh.elements[0].dofs[1, 0] == 6)

def test_continuous_dof_handler_total():
    msh = simple_line_mesh(4)
    bf = BasisFunctions.from_degree(1)
    apply_to_elements(msh, "basis", bf, non_gen = True)
    apply_to_elements(msh, "continuous", True, non_gen = True)
    total_dofs = init_dofs(msh)
    assert(total_dofs == 10)

def test_continuous_dof_handler_linear():
    msh = simple_line_mesh(4)
    bf = BasisFunctions.from_degree(1)
    apply_to_elements(msh, "basis", bf, non_gen = True)
    apply_to_elements(msh, "continuous", True, non_gen = True)
    total_dofs = init_dofs(msh)
    assert(msh.elements[0].dofs[0, 1] == 1)
    assert(msh.elements[1].dofs[0, 0] == 1)
    assert(msh.elements[2].dofs[1, 1] == 8)
    assert(msh.elements[3].dofs[1, 0] == 8)

def test_continuous_dof_handler_quadratic():
    msh = simple_line_mesh(2)
    bf = BasisFunctions.from_degree(2)
    apply_to_elements(msh, "basis", bf, non_gen = True)
    apply_to_elements(msh, "continuous", True, non_gen = True)
    total_dofs = init_dofs(msh)
    assert(msh.elements[0].dofs[0, 2] == 2)
    assert(msh.elements[1].dofs[0, 0] == 2)
    assert(msh.elements[1].dofs[0, 1] == 3)

def test_continuous_dof_handler_loop():
    vertices = np.array([(0.0, 1.0), (1.0, 0.0)])
    element_to_vertex = np.array([[0, 1], [1, 0]])
    m = from_vertices_and_etov(vertices, element_to_vertex)
    bf = BasisFunctions.from_degree(1)
    apply_to_elements(m, "basis", bf, non_gen = True)
    apply_to_elements(m, "continuous", True, non_gen = True)
    total_dofs = init_dofs(m)
    assert(m.elements[0].dofs[0, 0] == m.elements[1].dofs[0, 1])
    assert(m.elements[0].dofs[0, 1] == m.elements[1].dofs[0, 0])
