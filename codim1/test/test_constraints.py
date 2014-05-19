import numpy as np
from codim1.core import *
from codim1.assembly.constraints import apply_average_constraint, \
                                        pin_ends_constraint

def constraints_setup():
    msh = simple_line_mesh(2)
    bf = BasisFunctions.from_degree(1)
    apply_to_elements(msh, "basis", bf, non_gen = True)
    apply_to_elements(msh, "continuous", False, non_gen = True)
    init_dofs(msh)
    return msh

def test_avg_constraint():
    msh = constraints_setup()
    m = np.zeros((msh.total_dofs,msh.total_dofs))
    r = np.zeros(msh.total_dofs)
    apply_average_constraint(m, r, msh)
    np.testing.assert_almost_equal(m[0, :4], 0.5)
    np.testing.assert_almost_equal(m[4, 4:], 0.5)
    np.testing.assert_almost_equal(r, 0)

def test_pt_constraint():
    msh = constraints_setup()
    m = np.zeros((msh.total_dofs,msh.total_dofs))
    r = np.zeros(msh.total_dofs)
    pin_ends_constraint(m, r, msh, (1, 2), (3, 4))
    np.testing.assert_almost_equal(m[0, 0], 1.0)
    np.testing.assert_almost_equal(m[3, 3], 1.0)
    np.testing.assert_almost_equal(m[4, 4], 1.0)
    np.testing.assert_almost_equal(m[7, 7], 1.0)
    np.testing.assert_almost_equal(r, [1, 0, 0, 3, 2, 0, 0, 4])
