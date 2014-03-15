import numpy as np
from codim1.core.assembler import Assembler
import codim1.core.basis_funcs as basis_funcs
import codim1.fast.elastic_kernel as elastic_kernel
import codim1.core.mesh as mesh
import codim1.core.dof_handler as dof_handler
import codim1.core.tools as tools
import codim1.core.quad_strategy as quad_strategy

class TestKernel(object):
    """
    This class exists to assist with testing matrix assembly.
    The normal kernels are too complex to make testing easy.
    """
    def displacement_kernel(self, rx, ry, nx, ny):
        dist = np.sqrt(rx ** 2 + ry ** 2)
        return np.array([[np.log(1.0 / dist), 1.0], [1.0, np.log(1.0 / dist)]])

    def traction_kernel(self, rx, ry, nx, ny):
        return np.ones((2, 2))

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
    dh = dof_handler.DiscontinuousDOFHandler(msh, degree)
    bf = basis_funcs.BasisFunctions.from_degree(degree)
    k = TestKernel()
    assembler = Assembler(msh, bf, k, dh, qs)
    return assembler


def test_assemble_one_interaction_same_dof():
    a = simple_assembler(degree = 1)
    G_local, H_local, M_local = a.assemble_one_interaction(0, 0, 0, 0)
    # -0.5 * integral of (1-x)^2 from 0 to 1
    np.testing.assert_almost_equal(M_local, -(1.0 / 6.0))


def test_assemble_one_interaction_same_dof_with_jacobian():
    a = simple_assembler(degree = 1, n_elements = 4)
    G_local, H_local, M_local = a.assemble_one_interaction(0, 0, 0, 0)
    # Element size divided by two so the M value should be divided by two
    np.testing.assert_almost_equal(M_local, -(1.0 / 12.0))


def test_assemble_one_interaction_diff_dof():
    a = simple_assembler(degree = 1)
    G_local, H_local, M_local = a.assemble_one_interaction(0, 0, 0, 1)
    # -0.5 * integral of (1-x)*x from 0 to 1
    np.testing.assert_almost_equal(M_local, -(1.0 / 12.0))


def test_assemble_one_element_off_diagonal():
    a = simple_assembler(nonsing_pts = 10, logr_pts = 10, oneoverr_pts = 10)
    G_local, H_local, M_local = a.assemble_one_interaction(0, 0, 1, 0)
    np.testing.assert_almost_equal(H_local, np.ones((2, 2)))
    G_exact = np.array([[0.113706, 1.0],
                        [1.0, 0.113706]])
    np.testing.assert_almost_equal(G_local, G_exact, 4)


def test_assemble_one_element_on_diagonal():
    a = simple_assembler(nonsing_pts = 15, logr_pts = 16, oneoverr_pts = 10)
    G_local, H_local, M_local = a.assemble_one_interaction(0, 0, 0, 0)
    H_local[0, 0] += M_local
    H_local[1, 1] += M_local
    np.testing.assert_almost_equal(H_local, np.array([[0.5, 1.0], [1.0, 0.5]]))
    G_exact = np.array([[1.5, 1.0],
                        [1.0, 1.5]])
    np.testing.assert_almost_equal(G_local, G_exact, 4)


def test_assemble_row():
    a = simple_assembler(nonsing_pts = 16, logr_pts = 16, oneoverr_pts = 16)

    # The row functions should return one vector for each dimension.
    (G_row_x, G_row_y), (H_row_x, H_row_y)  = a.assemble_row(0, 0)

    np.testing.assert_almost_equal(H_row_x, np.array([0.5, 1.0, 1.0, 1.0]))
    np.testing.assert_almost_equal(H_row_y, np.array([1.0, 1.0, 0.5, 1.0]))

    # Haha, I made a pun.
    G_row_xact = np.array([1.5, 0.113706, 1.0, 1.0])
    G_row_yact = np.array([1.0, 1.0, 1.5, 0.113706])
    np.testing.assert_almost_equal(G_row_x, G_row_xact, 4)
    np.testing.assert_almost_equal(G_row_y, G_row_yact, 4)


def test_assemble():
    a = simple_assembler()
    H, G = a.assemble()
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
    H, G = a.assemble()
    np.testing.assert_almost_equal((H - H.T), np.zeros_like(H))
    np.testing.assert_almost_equal((G - G.T), np.zeros_like(G))


##
## Some more realistic assembly tests.
##
def realistic_assembler(n_elements = 4,
                        element_deg = 1,
                        quad_points_nonsingular = 5,
                        quad_points_logr = 5,
                        quad_points_oneoverr = 5,
                        left = -1.0,
                        right = 1.0):
    dim = 2
    shear_modulus = 1.0
    poisson_ratio = 0.25
    if quad_points_oneoverr % 2 == 1:
        quad_points_oneoverr += 1
    bf = basis_funcs.BasisFunctions.from_degree(element_deg)
    msh = mesh.Mesh.simple_line_mesh(n_elements, left, right)
    kernel = elastic_kernel.ElastostaticKernel(shear_modulus, poisson_ratio)
    dh = dof_handler.ContinuousDOFHandler(msh, element_deg)
    qs = quad_strategy.QuadStrategy(msh, quad_points_nonsingular,
                        quad_points_nonsingular,
                        quad_points_logr, quad_points_oneoverr)
    assembler = Assembler(msh, bf, kernel, dh, qs)
    return assembler


def test_exact_dbl_integrals_H_same_element():
    """
    This is probably the best test of working the H matrix is being
    assembled properly
    """
    a = realistic_assembler(quad_points_nonsingular = 10,
                            quad_points_logr = 10,
                            quad_points_oneoverr = 10,
                            n_elements = 1)
    H_00 = a.double_integral(a.kernel.traction_kernel, a.quad_strategy.quad_oneoverr,
                      0, 0, 0, 0)
    np.testing.assert_almost_equal(H_00, np.zeros((2, 2)), 3)

    H_11 = a.double_integral(a.kernel.traction_kernel, a.quad_strategy.quad_oneoverr,
                      0, 1, 0, 1)
    np.testing.assert_almost_equal(H_11, np.zeros((2, 2)), 3)

    H_01 = a.double_integral(a.kernel.traction_kernel, a.quad_strategy.quad_oneoverr,
                      0, 0, 0, 1)
    H_01_exact = np.array([[0.0, 1 / (6 * np.pi)],
                           [-1 / (6 * np.pi), 0.0]])
    np.testing.assert_almost_equal(H_01, H_01_exact, 3)

    H_10 = a.double_integral(a.kernel.traction_kernel, a.quad_strategy.quad_oneoverr,
                      0, 1, 0, 0)
    H_10_exact = np.array([[0.0, -1 / (6 * np.pi)],
                           [1 / (6 * np.pi), 0.0]])
    np.testing.assert_almost_equal(H_10, H_10_exact, 3)

def test_exact_dbl_integrals_G_same_element():
    """
    This is probably the best test of working the G matrix is being
    assembled properly
    """
    a = realistic_assembler(quad_points_nonsingular = 14,
                            quad_points_logr = 14,
                            quad_points_oneoverr = 10,
                            n_elements = 1,
                            left = -1.0,
                            right = 1.0)
    G_00 = a.double_integral(a.kernel.displacement_kernel, a.quad_strategy.quad_logr,
                      0, 0, 0, 0)
    np.testing.assert_almost_equal(G_00, [[0.165187, 0], [0, 0.112136]], 4)

    G_10 = a.double_integral(a.kernel.displacement_kernel, a.quad_strategy.quad_logr,
                      0, 1, 0, 0)
    np.testing.assert_almost_equal(G_10, [[0.112136, 0], [0, 0.0590839]], 4)
    G_01 = a.double_integral(a.kernel.displacement_kernel, a.quad_strategy.quad_logr,
                      0, 0, 0, 1)
    np.testing.assert_almost_equal(G_01, [[0.112136, 0], [0, 0.0590839]], 4)
    G_11 = a.double_integral(a.kernel.displacement_kernel, a.quad_strategy.quad_logr,
                      0, 1, 0, 1)
    np.testing.assert_almost_equal(G_11, [[0.165187, 0], [0, 0.112136]], 4)

def test_exact_dbl_integrals_G_different_element():
    """
    This is probably the best test of working the G matrix is being
    assembled properly
    """
    a = realistic_assembler(quad_points_nonsingular = 14,
                            quad_points_logr = 14,
                            quad_points_oneoverr = 10,
                            n_elements = 2,
                            left = -1.0,
                            right = 1.0)
    q = [a.quad_strategy.quad_shared_edge_left] * \
            len(a.quad_strategy.get_simple().x)
    G_00 = a.double_integral(a.kernel.displacement_kernel, q, 0, 0, 1, 0)
    np.testing.assert_almost_equal(G_00, [[0.0150739, 0], [0, 0.00181103]], 4)
    G_10 = a.double_integral(a.kernel.displacement_kernel, q, 0, 1, 1, 0)
    np.testing.assert_almost_equal(G_10, [[0.02833119, 0], [0, 0.01506828]], 4)
    G_01 = a.double_integral(a.kernel.displacement_kernel, q, 0, 0, 1, 1)
    np.testing.assert_almost_equal(G_01, [[0.00663146, 0], [0, -0.00663146]], 4)
    G_11 = a.double_integral(a.kernel.displacement_kernel, q, 0, 1, 1, 1)
    np.testing.assert_almost_equal(G_11, [[0.0150739, 0], [0, 0.00181103]], 4)

def test_realistic_nan():
    a = realistic_assembler()
    H, G = a.assemble()
    assert(not np.isnan(np.sum(H)))
    assert(not np.isnan(np.sum(G)))


def test_realistic_symmetric_linear():
    a = realistic_assembler()
    H, G = a.assemble()
    np.testing.assert_almost_equal((G - G.T) / np.mean(G), np.zeros_like(G))


def test_realistic_double_integral_symmetry():
    a = realistic_assembler(n_elements = 2, element_deg = 1)
    # fnc = lambda r, n: 1 / r[0]
    fnc = a.kernel.displacement_kernel
    one = a.double_integral(fnc,
                      a.quad_strategy.quad_oneoverr,
                      1, 0, 1, 1)

    two = a.double_integral(fnc,
                      a.quad_strategy.quad_oneoverr,
                      1, 1, 1, 0)
    np.testing.assert_almost_equal(one, two)


def test_realistic_symmetric_quadratic():
    a = realistic_assembler(quad_points_nonsingular = 14,
                            quad_points_logr = 14,
                            quad_points_oneoverr = 10,
                            n_elements = 1, element_deg = 2)
    H, G = a.assemble()
    np.testing.assert_almost_equal((G - G.T) / np.mean(G), np.zeros_like(G))


def test_reciprocal_effects():
    a = realistic_assembler(n_elements = 2)
    H, G = a.assemble()
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
    H, G = a.assemble()
    fnc = lambda x: (0.0, 1.0)
    displacements = tools.interpolate(fnc, a.dof_handler,
                                  a.basis_funcs, a.mesh)
    rhs = np.dot(H, displacements)
    soln = np.linalg.solve(G, rhs)
    for k in range(a.mesh.n_elements - 1):
        value_left = tools.evaluate_solution_on_element(k, 1.0, soln,
            a.dof_handler, a.basis_funcs, a.mesh)
        value_right = tools.evaluate_solution_on_element(k + 1, 0.0, soln,
            a.dof_handler, a.basis_funcs, a.mesh)
        np.testing.assert_almost_equal(value_left, value_right)