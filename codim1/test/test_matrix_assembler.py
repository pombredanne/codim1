import numpy as np
from codim1.core.matrix_assembler import MatrixAssembler
import codim1.core.basis_funcs as basis_funcs
from codim1.fast_lib import Kernel, DisplacementKernel, TractionKernel
import codim1.core.mesh as mesh
import codim1.core.dof_handler as dof_handler
import codim1.core.tools as tools
import codim1.core.quad_strategy as quad_strategy

class TDispKernel(Kernel):
    """
    This class exists to assist with testing matrix assembly.
    The normal kernels are too complex to make testing easy.
    """
    def __init__(self):
        self.singularity_type = 'logr'
        super(TDispKernel, self).__init__()

    def _call(self, data, p, q):
        if p == q:
            return np.log(1.0 / data.dist)
        return 1.0

class TTracKernel(Kernel):
    def __init__(self):
        self.singularity_type = 'oneoverr'
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
    msh = mesh.Mesh.simple_line_mesh(n_elements)
    qs = quad_strategy.QuadStrategy(msh, nonsing_pts, nonsing_pts,
                     logr_pts, oneoverr_pts)
    bf = basis_funcs.BasisFunctions.from_degree(degree)
    dh = dof_handler.DOFHandler(msh, bf, range(n_elements))
    assembler = MatrixAssembler(msh, bf, dh, qs)
    return assembler

def test_assemble_one_element_off_diagonal():
    a = simple_assembler(nonsing_pts = 10, logr_pts = 10, oneoverr_pts = 10)
    k_d = TDispKernel()
    k_t = TTracKernel()
    G_local = a.assemble_one_interaction(k_d, 0, 0, 1, 0)
    H_local = a.assemble_one_interaction(k_t, 0, 0, 1, 0)
    np.testing.assert_almost_equal(H_local, np.ones((2, 2)))
    np.testing.assert_almost_equal(G_local,
            np.array([[0.113706, 1.0], [1.0, 0.113706]]), 4)


def test_assemble_one_element_on_diagonal():
    a = simple_assembler(nonsing_pts = 12, logr_pts = 17, oneoverr_pts = 10)
    k_d = TDispKernel()
    k_t = TTracKernel()
    G_local = a.assemble_one_interaction(k_d, 0, 0, 0, 0)
    H_local = a.assemble_one_interaction(k_t, 0, 0, 0, 0)
    np.testing.assert_almost_equal(H_local,
                                   np.array([[1.0, 1.0], [1.0, 1.0]]))
    np.testing.assert_almost_equal(G_local,
                                   np.array([[1.5, 1.0], [1.0, 1.5]]), 4)


def test_assemble_row():
    a = simple_assembler(nonsing_pts = 16, logr_pts = 16, oneoverr_pts = 16)
    k_d = TDispKernel()
    k_t = TTracKernel()

    # The row functions should return one vector for each dimension.
    (G_row_x, G_row_y) = a.assemble_row(k_d, 0, 0)
    (H_row_x, H_row_y) = a.assemble_row(k_t, 0, 0)

    np.testing.assert_almost_equal(H_row_x, np.array([1.0, 1.0, 1.0, 1.0]))
    np.testing.assert_almost_equal(H_row_y, np.array([1.0, 1.0, 1.0, 1.0]))

    # Haha, I made a pun.
    G_row_xact = np.array([1.5, 0.113706, 1.0, 1.0])
    G_row_yact = np.array([1.0, 1.0, 1.5, 0.113706])
    np.testing.assert_almost_equal(G_row_x, G_row_xact, 4)
    np.testing.assert_almost_equal(G_row_y, G_row_yact, 4)


def test_assemble():
    a = simple_assembler()
    k_d = TDispKernel()
    k_t = TTracKernel()
    G = a.assemble_matrix(k_d)
    H = a.assemble_matrix(k_t)
    # Just make sure it worked. Don't check for correctness.
    assert(H.shape[0] == a.dof_handler.total_dofs)
    assert(H.shape[1] == a.dof_handler.total_dofs)
    assert(G.shape[0] == a.dof_handler.total_dofs)
    assert(G.shape[1] == a.dof_handler.total_dofs)
    assert(not np.isnan(np.sum(H)))
    assert(not np.isnan(np.sum(G)))


def test_simple_symmetric_linear():
    # The test kernel is completely symmetric as are the basis functions.
    a = simple_assembler(n_elements = 1, degree = 1,
                         nonsing_pts = 4, logr_pts = 4, oneoverr_pts = 4)
    k_d = TDispKernel()
    k_t = TTracKernel()
    G = a.assemble_matrix(k_d)
    H = a.assemble_matrix(k_t)
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
    msh = mesh.Mesh.simple_line_mesh(n_elements, left, right)
    bf = basis_funcs.BasisFunctions.from_degree(element_deg)
    dh = dof_handler.DOFHandler(msh, bf)
    qs = quad_strategy.QuadStrategy(msh, quad_points_nonsingular,
                        quad_points_nonsingular,
                        quad_points_logr, quad_points_oneoverr)
    assembler = MatrixAssembler(msh, bf, dh, qs)
    return assembler


def test_realistic_nan():
    a = realistic_assembler()
    k_d = DisplacementKernel(1.0, 0.25)
    k_t = TractionKernel(1.0, 0.25)
    G = a.assemble_matrix(k_d)
    H = a.assemble_matrix(k_t)
    assert(not np.isnan(np.sum(H)))
    assert(not np.isnan(np.sum(G)))


def test_realistic_symmetric_linear():
    a = realistic_assembler()
    k_d = DisplacementKernel(1.0, 0.25)
    k_t = TractionKernel(1.0, 0.25)
    G = a.assemble_matrix(k_d)
    H = a.assemble_matrix(k_t)
    np.testing.assert_almost_equal((G - G.T) /
                                    np.mean(G), np.zeros_like(G), 4)


def test_realistic_symmetric_quadratic():
    a = realistic_assembler(quad_points_nonsingular = 10,
                            quad_points_logr = 12,
                            quad_points_oneoverr = 10,
                            n_elements = 1, element_deg = 2)
    k_d = DisplacementKernel(1.0, 0.25)
    k_t = TractionKernel(1.0, 0.25)
    G = a.assemble_matrix(k_d)
    H = a.assemble_matrix(k_t)
    np.testing.assert_almost_equal((G - G.T) / np.mean(G), np.zeros_like(G), 4)


def test_reciprocal_effects():
    a = realistic_assembler(n_elements = 2)
    k_d = DisplacementKernel(1.0, 0.25)
    k_t = TractionKernel(1.0, 0.25)
    G = a.assemble_matrix(k_d)
    H = a.assemble_matrix(k_t)
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
    a = realistic_assembler(element_deg = 1)
    k_d = DisplacementKernel(1.0, 0.25)
    k_t = TractionKernel(1.0, 0.25)
    G = a.assemble_matrix(k_d)
    H = a.assemble_matrix(k_t)
    fnc = lambda x, n: (0.0, 1.0)
    displacements = tools.interpolate(fnc, a.dof_handler,
                                  a.basis_funcs, a.mesh)
    rhs = np.dot(H, displacements)
    soln_coeffs = np.linalg.solve(G, rhs)
    soln = basis_funcs.Solution(a.basis_funcs, a.dof_handler, soln_coeffs)
    for k in range(a.mesh.n_elements - 1):
        value_left = tools.evaluate_solution_on_element(k, 1.0, soln, a.mesh)
        value_right = tools.evaluate_solution_on_element(k + 1, 0.0, soln,
                                                         a.mesh)
        np.testing.assert_almost_equal(value_left, value_right)
