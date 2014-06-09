import numpy as np
from codim1.assembly.matrix_assembler import \
    simple_matrix_assemble, _compute_one_interaction, _compute_element_pair
import codim1.core.basis_funcs as basis_funcs
from codim1.fast_lib import Kernel, DisplacementKernel, TractionKernel
import codim1.core.mesh as mesh
import codim1.core.dof_handler as dof_handler
from codim1.core import *
import codim1.core.tools as tools
import codim1.core.quad_strategy as quad_strategy
from codim1.post import *

class TDispKernel(Kernel):
    """
    This class exists to assist with testing matrix assembly.
    The normal kernels are too complex to make testing easy.
    """
    def __init__(self):
        self.singularity_type = 'logr'
        self.test_gradient = False
        self.soln_gradient = False
        super(TDispKernel, self).__init__()

    def _call(self, data, p, q):
        if p == q:
            return np.log(1.0 / data.dist)
        return 1.0

class TTracKernel(Kernel):
    def __init__(self):
        self.singularity_type = 'oneoverr'
        self.test_gradient = False
        self.soln_gradient = False
        super(TTracKernel, self).__init__()

    def _call(self, data, p, q):
        return 1.0

def simple_assembler(degree = 0,
                     nonsing_pts = 2,
                     logr_pts = 2,
                     oneoverr_pts = 2,
                     n_elements = 2):
    if oneoverr_pts % 2 == 1:
        oneoverr_pts += 1
    msh = simple_line_mesh(n_elements)
    qs = quad_strategy.QuadStrategy(msh, nonsing_pts, nonsing_pts,
                     logr_pts, oneoverr_pts)
    bf = basis_funcs.basis_from_degree(degree)
    apply_to_elements(msh, "basis", bf, non_gen = True)
    apply_to_elements(msh, "continuous", False, non_gen = True)
    apply_to_elements(msh, "qs", qs, non_gen = True)
    init_dofs(msh)
    return msh

def test_assemble_one_element_off_diagonal():
    msh = simple_assembler(nonsing_pts = 10, logr_pts = 10, oneoverr_pts = 10)
    k_d = TDispKernel()
    k_t = TTracKernel()
    G_local = _compute_one_interaction(k_d, msh.elements[0], 0,
                                      msh.elements[1], 0)
    H_local = _compute_one_interaction(k_t, msh.elements[0], 0,
                                      msh.elements[1], 0)
    np.testing.assert_almost_equal(H_local, np.ones((2, 2)))
    np.testing.assert_almost_equal(G_local,
            np.array([[0.113706, 1.0], [1.0, 0.113706]]), 4)


def test_assemble_one_element_on_diagonal():
    msh = simple_assembler(nonsing_pts = 12, logr_pts = 17, oneoverr_pts = 10)
    k_d = TDispKernel()
    k_t = TTracKernel()
    G_local = _compute_one_interaction(k_d, msh.elements[0], 0,
                                      msh.elements[0], 0)
    H_local = _compute_one_interaction(k_t, msh.elements[0], 0,
                                      msh.elements[0], 0)
    np.testing.assert_almost_equal(H_local,
                                   np.array([[1.0, 1.0], [1.0, 1.0]]))
    np.testing.assert_almost_equal(G_local,
                                   np.array([[1.5, 1.0], [1.0, 1.5]]), 4)


def test_assemble():
    msh = simple_assembler()
    k_d = TDispKernel()
    k_t = TTracKernel()
    G = simple_matrix_assemble(msh, k_d)
    H = simple_matrix_assemble(msh, k_t)
    # Just make sure it worked. Don't check for correctness.
    assert(H.shape[0] == msh.total_dofs)
    assert(H.shape[1] == msh.total_dofs)
    assert(G.shape[0] == msh.total_dofs)
    assert(G.shape[1] == msh.total_dofs)
    assert(not np.isnan(np.sum(H)))
    assert(not np.isnan(np.sum(G)))


def test_simple_symmetric_linear():
    # The test kernel is completely symmetric as are the basis functions.
    msh = simple_assembler(n_elements = 1, degree = 1,
                         nonsing_pts = 4, logr_pts = 4, oneoverr_pts = 4)
    k_d = TDispKernel()
    k_t = TTracKernel()
    G = simple_matrix_assemble(msh, k_d)
    H = simple_matrix_assemble(msh, k_t)
    np.testing.assert_almost_equal((H - H.T), np.zeros_like(H))
    np.testing.assert_almost_equal((G - G.T), np.zeros_like(G))

##
## Some more realistic assembly tests.
##
def realistic_assembler(n_elements = 4,
                        element_deg = 1,
                        quad_points_nonsingular = 5,
                        quad_points_logr = 6,
                        quad_points_oneoverr = 5,
                        left = -1.0,
                        right = 1.0):
    dim = 2
    if quad_points_oneoverr % 2 == 1:
        quad_points_oneoverr += 1
    msh = simple_line_mesh(n_elements, (left, 0.0), (right, 0.0))
    bf = basis_funcs.basis_from_degree(element_deg)
    qs = quad_strategy.QuadStrategy(msh, quad_points_nonsingular,
                        quad_points_nonsingular,
                        quad_points_logr, quad_points_oneoverr)
    apply_to_elements(msh, "basis", bf, non_gen = True)
    apply_to_elements(msh, "continuous", True, non_gen = True)
    apply_to_elements(msh, "qs", qs, non_gen = True)
    init_dofs(msh)
    return msh


def test_realistic_nan():
    msh = realistic_assembler()
    k_d = DisplacementKernel(1.0, 0.25)
    k_t = TractionKernel(1.0, 0.25)
    G = simple_matrix_assemble(msh, k_d)
    H = simple_matrix_assemble(msh, k_t)
    assert(not np.isnan(np.sum(H)))
    assert(not np.isnan(np.sum(G)))


def test_realistic_symmetric_linear():
    msh = realistic_assembler()
    k_d = DisplacementKernel(1.0, 0.25)
    k_t = TractionKernel(1.0, 0.25)
    G = simple_matrix_assemble(msh, k_d)
    H = simple_matrix_assemble(msh, k_t)
    np.testing.assert_almost_equal((G - G.T) /
                                    np.mean(G), np.zeros_like(G), 4)


def test_realistic_symmetric_quadratic():
    msh = realistic_assembler(quad_points_nonsingular = 10,
                            quad_points_logr = 12,
                            quad_points_oneoverr = 10,
                            n_elements = 1, element_deg = 2)
    k_d = DisplacementKernel(1.0, 0.25)
    k_t = TractionKernel(1.0, 0.25)
    G = simple_matrix_assemble(msh, k_d)
    H = simple_matrix_assemble(msh, k_t)
    np.testing.assert_almost_equal((G - G.T) / np.mean(G), np.zeros_like(G), 4)


def test_reciprocal_effects():
    msh = realistic_assembler(n_elements = 2)
    k_d = DisplacementKernel(1.0, 0.25)
    k_t = TractionKernel(1.0, 0.25)
    G = simple_matrix_assemble(msh, k_d)
    H = simple_matrix_assemble(msh, k_t)
    # The influence of u_x(0) on u_y(1) should be the opposite of the
    # effect of u_x(1) on u_y(0), where the parenthesis indicate which element
    np.testing.assert_almost_equal(H[4,0], -H[3,1], 2)
    # The influence of u_y(0) on u_x(1) should be the opposite of the
    # effect of u_y(1) on u_x(0), where the parenthesis indicate which element
    # Really just the symmetric part of the above...
    np.testing.assert_almost_equal(H[5,1], -H[4,2], 2)
    # They should be equal for G
    np.testing.assert_almost_equal(G[4,0], G[3,1])


def test_realistic_zero_discontinuity():
    msh = realistic_assembler(element_deg = 1)
    k_d = DisplacementKernel(1.0, 0.25)
    k_t = TractionKernel(1.0, 0.25)
    G = simple_matrix_assemble(msh, k_d)
    H = simple_matrix_assemble(msh, k_t)
    fnc = lambda x, n: (0.0, 1.0)
    displacements = tools.interpolate(fnc, msh)
    rhs = np.dot(H, displacements)
    soln_coeffs = np.linalg.solve(G, rhs)
    apply_to_elements(msh, "bc", BC("traction", msh.elements[0].basis),
                      non_gen = True)
    for k in range(msh.n_elements - 1):
        e_k = msh.elements[k]
        e_kp1 = msh.elements[k + 1]
        value_left,t = evaluate_solution_on_element(e_k, 1.0, soln_coeffs)
        value_right,t = evaluate_solution_on_element(e_kp1, 0.0, soln_coeffs)
        np.testing.assert_almost_equal(value_left, value_right)
