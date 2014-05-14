
import numpy as np
from codim1.core import *

def test_avg_constraint():
    msh = simple_line_mesh(2)
    bf = BasisFunctions.from_degree(1)
    dh = DOFHandler(msh, bf, range(2))
    m = np.zeros((dh.total_dofs,dh.total_dofs))
    r = np.zeros(dh.total_dofs)
    apply_average_constraint(m, r, msh, bf, dh)
    np.testing.assert_almost_equal(m[0, :4], 0.5)
    np.testing.assert_almost_equal(m[4, 4:], 0.5)
    np.testing.assert_almost_equal(r, 0)

def test_pt_constraint():
    msh = simple_line_mesh(2)
    bf = BasisFunctions.from_degree(1)
    dh = DOFHandler(msh, bf, range(2))
    m = np.zeros((dh.total_dofs,dh.total_dofs))
    r = np.zeros(dh.total_dofs)
    pin_ends_constraint(m, r, (1, 2), (3, 4), dh)
    np.testing.assert_almost_equal(m[0, 0], 1.0)
    np.testing.assert_almost_equal(m[3, 3], 1.0)
    np.testing.assert_almost_equal(m[4, 4], 1.0)
    np.testing.assert_almost_equal(m[7, 7], 1.0)
    np.testing.assert_almost_equal(r, [1, 0, 0, 3, 2, 0, 0, 4])
