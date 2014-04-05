import numpy as np
from codim1.core.dof_handler import DOFHandler
from codim1.core.basis_funcs import BasisFunctions
import codim1.core.mesh as mesh

def test_mixed_dof_handler():
    msh = mesh.Mesh.simple_line_mesh(3)
    bf = BasisFunctions.from_degree(2)
    discontinuous_elements = [2]
    dh = DOFHandler(msh, bf, discontinuous_elements)
    assert(dh.dof_map[0, 0, 0] == 0)
    assert(dh.dof_map[0, 1, 0] == 2)
    assert(dh.dof_map[0, 2, 0] == 5)
    assert(dh.dof_map[0, 2, 1] == 6)

def test_discontinuous_dof_handler():
    msh = mesh.Mesh.simple_line_mesh(2)
    bf = BasisFunctions.from_degree(2)
    dh = DOFHandler(msh, bf, [0, 1])
    assert(dh.dof_map[0, 0, 0] == 0)
    assert(dh.dof_map[0, 1, 1] == 4)
    assert(dh.dof_map[1, 0, 0] == 6)

def test_continuous_dof_handler_total():
    msh = mesh.Mesh.simple_line_mesh(4)
    bf = BasisFunctions.from_degree(1)
    dh = DOFHandler(msh, bf)
    assert(dh.total_dofs == 10)

def test_continuous_dof_handler_linear():
    msh = mesh.Mesh.simple_line_mesh(4)
    bf = BasisFunctions.from_degree(1)
    dh = DOFHandler(msh, bf)
    assert(dh.dof_map[0, 0, 1] == 1)
    assert(dh.dof_map[0, 1, 0] == 1)
    assert(dh.dof_map[1, 2, 1] == 8)
    assert(dh.dof_map[1, 3, 0] == 8)

def test_continuous_dof_handler_quadratic():
    msh = mesh.Mesh.simple_line_mesh(2)
    bf = BasisFunctions.from_degree(2)
    dh = DOFHandler(msh, bf)
    assert(dh.dof_map[0, 0, 2] == 2)
    assert(dh.dof_map[0, 1, 0] == 2)
    assert(dh.dof_map[0, 1, 1] == 3)

def test_continuous_dof_handler_loop():
    vp = np.array([0.0, 1.0])
    vp_func = lambda x: np.array([x, 1.0 - x])
    element_to_vertex = np.array([[0, 1], [1, 0]])
    m = mesh.Mesh(vp_func, vp, element_to_vertex)
    bf = BasisFunctions.from_degree(1)
    dh = DOFHandler(m, bf)
    assert(dh.dof_map[0, 0, 0] == dh.dof_map[0, 1, 1])
    assert(dh.dof_map[0, 0, 1] == dh.dof_map[0, 1, 0])
