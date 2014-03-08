import quadrature
import numpy as np

class Assembler(object):
    """
    This class builds the matrices needed for a boundary element method.
    Two matrices are created. The terminology here is in reference to the
    derivation based on the reciprocal theorem. The definitions of these
    matrices can be derived by performing integration by parts on the
    original governing equations of static elasticity.

    1. H -- The traction kernel matrix represents the test tractions acting
    through the solution displacements at each node of each element.

    2. G -- The displacement kernel matrix represents the test displacements
    acting through the solution tractions at each node of each element.
    """
    def __init__(self,
                 mesh,
                 basis_funcs,
                 kernel,
                 dof_handler,
                 quad_points_nonsingular,
                 quad_points_logr,
                 quad_points_oneoverr):
        self.mesh = mesh
        self.basis_funcs = basis_funcs
        self.kernel = kernel
        self.dof_handler = dof_handler
        self.quad_points_nonsingular = quad_points_nonsingular
        self.quad_points_logr = quad_points_logr
        self.quad_points_oneoverr = quad_points_oneoverr

        self.setup_quadrature()

    def setup_quadrature(self):
        """
        The quadrature rules can be defined once on the reference element
        and then a change of variables allows integration on any element.
        """
        self.quad_nonsingular = quadrature.QuadGauss(
                self.quad_points_nonsingular)
        self.quad_logr = []
        self.quad_oneoverr = []
        for singular_pt in self.quad_nonsingular.x:
            logr = quadrature.QuadSingularTelles(self.quad_points_logr,
                                                 singular_pt)
            oneoverr = quadrature.QuadOneOverR(self.quad_points_oneoverr,
                                               singular_pt,
                                               self.quad_points_nonsingular)
            self.quad_logr.append(logr)
            self.quad_oneoverr.append(oneoverr)

    def assemble(self):
        """
        Produces the H and G matrices as described in the class description.
        Return (H, G)
        """
        total_dofs = self.dof_handler.total_dofs
        H = np.zeros((total_dofs, total_dofs))
        G = np.zeros((total_dofs, total_dofs))
        for k in range(self.mesh.n_elements):
            for i in range(self.basis_funcs.num_fncs):
                dof_x = self.dof_handler.dof_map[0, k, i]
                dof_y = self.dof_handler.dof_map[1, k, i]

                H_row_x, H_row_y = self.assemble_H_row(k, i)
                H[dof_x, :] += H_row_x
                H[dof_y, :] += H_row_y

                G_row_x, G_row_y = self.assemble_G_row(k, i)
                G[dof_x, :] += G_row_x
                G[dof_y, :] += G_row_y
        # Enforce the symmetry of G. Inefficient way of doing this, but it
        # works. The assymetry only results from small numerical errors in the
        # computation, so this statement shouldn't make the error any worse.
        assert(((G - G.T) / np.max(G) < 1e-4).all())
        G -= 0.5 * (G - G.T)
        return H, G

    def assemble_G_row(self, k, i):
        """
        Assemble one row of the G matrix.
        """
        G_row_x = np.zeros(self.dof_handler.total_dofs)
        G_row_y = np.zeros(self.dof_handler.total_dofs)
        for l in range(self.mesh.n_elements):
            for j in range(self.basis_funcs.num_fncs):
                soln_dof_x = self.dof_handler.dof_map[0, l, j]
                soln_dof_y = self.dof_handler.dof_map[1, l, j]

                G_local = self.assemble_G_one_interaction(k, i, l, j)
                # Add the local interactions to the global matrix in
                # the proper locations
                G_row_x[soln_dof_x] += G_local[0, 0]
                G_row_x[soln_dof_y] += G_local[0, 1]
                G_row_y[soln_dof_x] += G_local[1, 0]
                G_row_y[soln_dof_y] += G_local[1, 1]
        return G_row_x, G_row_y


    def assemble_H_row(self, k, i):
        """
        Assemble one row of the H matrix.
        """
        H_row_x = np.zeros(self.dof_handler.total_dofs)
        H_row_y = np.zeros(self.dof_handler.total_dofs)
        for l in range(self.mesh.n_elements):
            for j in range(self.basis_funcs.num_fncs):
                soln_dof_x = self.dof_handler.dof_map[0, l, j]
                soln_dof_y = self.dof_handler.dof_map[1, l, j]

                H_local = self.assemble_H_one_interaction(k, i, l, j)

                if k == l:
                    M_local = self.assemble_M_one_interaction(k, i, j)
                    # M_local is only applied on the block diagonal
                    H_local[0, 0] += M_local
                    H_local[1, 1] += M_local

                # import ipdb; ipdb.set_trace()
                # Add the local interactions to the global matrix in
                # the proper locations
                H_row_x[soln_dof_x] += H_local[0, 0]
                H_row_x[soln_dof_y] += H_local[0, 1]
                H_row_y[soln_dof_x] += H_local[1, 0]
                H_row_y[soln_dof_y] += H_local[1, 1]
        return H_row_x, H_row_y

    def assemble_G_one_interaction(self, k, i, l, j):
        """
        Compute one pair of element interactions for the G matrix.
        """
        G_sing = np.zeros((2, 2))
        quad = self.quad_nonsingular
        singular = False
        if k == l:
            quad = self.quad_logr
            singular = True
        G_local = self.double_integral(self.kernel.displacement_kernel,
                             G_sing, singular, quad,
                             k, i, l, j)
        return G_local

    def assemble_M_one_interaction(self, k, i, j):
        """
        Compute one local mass matrix interaction.
        """
        M_local = self.single_integral(lambda x: 1.0, 0.0, k, i, j)
        return -0.5 * M_local

    def assemble_H_one_interaction(self, k, i, l, j):
        """
        Compute one pair of element interactions for the H matrix.
        """
        H_local = np.zeros((2, 2))
        quad = self.quad_nonsingular
        singular = False
        if k == l:
            quad = self.quad_oneoverr
            singular = True
        H_local = self.double_integral(self.kernel.traction_kernel,
                             H_local, singular, quad,
                             k, i, l, j)
        return H_local

    def single_integral(self, kernel, result, k, i, j):
        """
        Performs a single integral over the element specified by k
        with the basis functions specified by i and j. q_pts and w
        define the quadrature rule. kernel should be a function that
        can be evaluated at all point within the element
        """
        jacobian = self.mesh.get_element_jacobian(k)
        q_pts = self.quad_nonsingular.x
        w = self.quad_nonsingular.w
        for (q_pt, w) in zip(q_pts, w):
            # The basis functions should be evaluated on reference
            # coordinates
            src_basis_fnc = self.basis_funcs.evaluate_basis(i, q_pt)
            soln_basis_fnc = self.basis_funcs.evaluate_basis(j, q_pt)
            phys_pt = self.mesh.get_physical_points(k, q_pt)[0]
            result += kernel(phys_pt) * \
                soln_basis_fnc * src_basis_fnc * jacobian * w
        return result

    def double_integral(self, kernel, result, singular, q_soln, k, i, l, j):
        """
        Performs a double integral over a pair of elements with the
        provided quadrature rule.

        Warning: This function modifies the "result" input.
        """
        # Jacobian determinants are necessary to scale the integral with the
        # change of variables.
        src_jacobian = self.mesh.get_element_jacobian(k)
        soln_jacobian = self.mesh.get_element_jacobian(l)

        # The normal is the one on the soln integration element.
        # This is clear if you remember the source is actually a point
        # and thus has no defined normal. We are integrating over many point
        # sources.
        normal = self.mesh.normals[l]

        # The outer quadrature uses a standard nonsingular quadrature formula
        q_pts = self.quad_nonsingular.x
        w = self.quad_nonsingular.w
        for (q_src_pt_index, (q_pt_src, w_src)) in enumerate(zip(q_pts, w)):
            phys_src_pt = self.mesh.get_physical_points(k, q_pt_src)
            # The basis functions should be evaluated on reference
            # coordinates
            src_basis_fnc = self.basis_funcs.evaluate_basis(i, q_pt_src)

            # If the integrand is singular, we need to use the appropriate
            # inner quadrature method. Which points the inner quadrature
            # chooses will depend on the current outer quadrature point
            # which will be the point of singularity, assuming same element
            if singular:
                q_pts_soln = q_soln[q_src_pt_index].x
                q_w_soln = q_soln[q_src_pt_index].w
                assert(q_soln[q_src_pt_index].x0 == q_pt_src)
            else:
                q_pts_soln = q_soln.x
                q_w_soln = q_soln.w

            for (q_pt_soln, w_soln) in zip(q_pts_soln, q_w_soln):
                soln_basis_fnc = self.basis_funcs.evaluate_basis(j, q_pt_soln)

                # Separation of the two quadrature points, use real,
                # physical coordinates!
                phys_soln_pt = self.mesh.get_physical_points(l, q_pt_soln)

                # From source to solution.
                r = phys_soln_pt - phys_src_pt

                # Actually evaluate the kernel.
                k_val = kernel(r, normal)
                assert(not np.isnan(np.sum(k_val))), \
                       "nan kernel value for R = " + str(np.linalg.norm(r))

                # Actually perform the quadrature
                result += k_val * src_basis_fnc * soln_basis_fnc *\
                          src_jacobian * soln_jacobian *\
                          w_soln * w_src
        return result


################################################################################
# TESTS                                                                        #
################################################################################

import basis_funcs
import elastic_kernel
import mesh
import dof_handler
import tools

class TestKernel(object):
    """
    This class exists to assist with testing matrix assembly.
    The normal kernels are too complex to make testing easy.
    """

    def displacement_singular(self, r, n):
        dist = np.sqrt(r[0] ** 2 + r[1] ** 2)
        return np.array([[np.log(dist), 0.0], [0.0, np.log(dist)]])
        # return 1.0 #np.array([[np.log(dist), 0.0], [0.0, np.log(dist)]])

    def displacement_nonsingular(self, r, n):
        return np.ones((2, 2))

    def displacement_kernel(self, r, n):
        return np.ones((2, 2))

    def traction_kernel(self, r, n):
        return np.ones((2, 2))

def simple_assembler(degree = 0,
                     nonsing_pts = 2,
                     logr_pts = 2,
                     oneoverr_pts = 2,
                     n_elements = 2):
    if oneoverr_pts % 2 == 1:
        oneoverr_pts += 1
    msh = mesh.Mesh.simple_line_mesh(n_elements)
    dh = dof_handler.DiscontinuousDOFHandler(msh, degree)
    bf = basis_funcs.BasisFunctions.from_degree(degree)
    k = TestKernel()
    assembler = Assembler(msh, bf, k, dh,
                          nonsing_pts, logr_pts, oneoverr_pts)
    return assembler

def test_build_quadrature_list():
    a = simple_assembler(degree = 2)

    assert(a.quad_nonsingular.N == 2)

    assert(len(a.quad_logr) == 2)
    assert(len(a.quad_oneoverr) == 2)

    assert(a.quad_logr[0].N == 2)
    assert(a.quad_oneoverr[0].N == 2)

    assert(a.quad_logr[0].x0 == a.quad_nonsingular.x[0])
    assert(a.quad_logr[1].x0 == a.quad_nonsingular.x[1])
    assert(a.quad_oneoverr[0].x0 == a.quad_nonsingular.x[0])
    assert(a.quad_oneoverr[1].x0 == a.quad_nonsingular.x[1])

def test_assemble_M_same_dof():
    a = simple_assembler(degree = 1)
    M_local = a.assemble_M_one_interaction(0, 0, 0)
    # -0.5 * integral of (1-x)^2 from 0 to 1
    np.testing.assert_almost_equal(M_local, -(1.0 / 6.0))

def test_assemble_M_same_dof_with_jacobian():
    a = simple_assembler(degree = 1, n_elements = 4)
    M_local = a.assemble_M_one_interaction(0, 0, 0)
    # Element size divided by two so the M value should be divided by two
    np.testing.assert_almost_equal(M_local, -(1.0 / 12.0))

def test_assemble_M_diff_dof():
    a = simple_assembler(degree = 1)
    M_local = a.assemble_M_one_interaction(0, 0, 1)
    # -0.5 * integral of (1-x)*x from 0 to 1
    np.testing.assert_almost_equal(M_local, -(1.0 / 12.0))

def test_assemble_H_one_element_off_diagonal():
    a = simple_assembler(oneoverr_pts = 2)
    H_local = a.assemble_H_one_interaction(0, 0, 1, 0)
    np.testing.assert_almost_equal(H_local, np.ones((2,2)))

def test_assemble_H_one_element_on_diagonal():
    a = simple_assembler(oneoverr_pts = 2)
    H_local = a.assemble_H_one_interaction(0, 0, 0, 0)
    np.testing.assert_almost_equal(H_local, np.ones((2,2)))

def test_assemble_H_row_test_kernel():
    a = simple_assembler(oneoverr_pts = 4)

    # The row functions should return one vector for each dimension.
    H_row_x, H_row_y = a.assemble_H_row(0, 0)

    np.testing.assert_almost_equal(H_row_x, np.array([0.5, 1.0, 1.0, 1.0]))
    np.testing.assert_almost_equal(H_row_y, np.array([1.0, 1.0, 0.5, 1.0]))

def test_assemble_G_one_element_off_diagonal():
    a = simple_assembler(nonsing_pts = 10, logr_pts = 2)
    G_local = a.assemble_G_one_interaction(0, 0, 1, 0)
    G_exact = np.array([[1.0 - 1.5 + np.log(4), 1.0],
                        [1.0, 1.0 - 1.5 + np.log(4)]])
    np.testing.assert_almost_equal(G_local, G_exact, 4)

def test_assemble_G_one_element_on_diagonal():
    a = simple_assembler(nonsing_pts = 8, logr_pts = 4)
    G_local = a.assemble_G_one_interaction(0, 0, 0, 0)
    G_exact = np.array([[1.0 - 1.5, 1.0],
                        [1.0, 1.0 - 1.5]])
    np.testing.assert_almost_equal(G_local, G_exact, 4)

def test_assemble_G_row_test_kernel():
    a = simple_assembler(nonsing_pts = 10, logr_pts = 4)

    # The row functions should return one vector for each dimension.
    G_row_x, G_row_y = a.assemble_G_row(0, 0)
    # Haha, I made a pun.
    G_row_xact = np.array([-0.5, 1.0 - 1.5 + np.log(4), 1.0, 1.0])
    G_row_yact = np.array([1.0, 1.0, -0.5, 1.0 - 1.5 + np.log(4)])

    np.testing.assert_almost_equal(G_row_x, G_row_xact, 4)
    np.testing.assert_almost_equal(G_row_y, G_row_yact, 4)

def test_assemble():
    a = simple_assembler()
    H, G = a.assemble()
    assert(H.shape[0] == a.dof_handler.total_dofs)
    assert(H.shape[1] == a.dof_handler.total_dofs)
    assert(G.shape[0] == a.dof_handler.total_dofs)
    assert(G.shape[1] == a.dof_handler.total_dofs)
    assert(not np.isnan(np.sum(H)))
    assert(not np.isnan(np.sum(G)))

def test_simple_symmetric_linear():
    a = simple_assembler(n_elements = 1, degree = 1,
                         nonsing_pts = 4, logr_pts = 4)
    H, G = a.assemble()
    np.testing.assert_almost_equal((H - H.T) / np.mean(H), np.zeros_like(H))
    np.testing.assert_almost_equal((G - G.T) / np.mean(G), np.zeros_like(G))

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
    assembler = Assembler(msh, bf, kernel, dh,
                          quad_points_nonsingular,
                          quad_points_logr,
                          quad_points_oneoverr)
    return assembler

def test_exact_dbl_integrals_H():
    """
    This is probably the best test of working the H matrix is being
    assembled properly
    """
    a = realistic_assembler(quad_points_nonsingular = 10,
                            quad_points_logr = 10,
                            quad_points_oneoverr = 10,
                            n_elements = 1)
    H_00 = a.double_integral(a.kernel.traction_kernel, np.zeros((2, 2)),
                      True, a.quad_oneoverr,
                      0, 0, 0, 0)
    np.testing.assert_almost_equal(H_00, np.zeros((2, 2)), 3)
    H_11 = a.double_integral(a.kernel.traction_kernel, np.zeros((2, 2)),
                      True, a.quad_oneoverr,
                      0, 1, 0, 1)
    np.testing.assert_almost_equal(H_11, np.zeros((2, 2)), 3)

    H_01 = a.double_integral(a.kernel.traction_kernel, np.zeros((2, 2)),
                      True, a.quad_oneoverr,
                      0, 0, 0, 1)
    H_01_exact = np.array([[0.0, 1 / (6 * np.pi)],
                           [-1 / (6 * np.pi), 0.0]])
    np.testing.assert_almost_equal(H_01, H_01_exact, 3)

    H_10 = a.double_integral(a.kernel.traction_kernel, np.zeros((2, 2)),
                      True, a.quad_oneoverr,
                      0, 1, 0, 0)
    H_10_exact = np.array([[0.0, -1 / (6 * np.pi)],
                           [1 / (6 * np.pi), 0.0]])
    np.testing.assert_almost_equal(H_10, H_10_exact, 3)


def test_exact_dbl_integrals_G():
    """
    This is probably the best test of working the G matrix is being
    assembled properly
    """
    a = realistic_assembler(quad_points_nonsingular = 10,
                            quad_points_logr = 10,
                            quad_points_oneoverr = 10,
                            n_elements = 1,
                            left = -1.0,
                            right = 1.0)
    G_00 = a.double_integral(a.kernel.displacement_singular, np.zeros((2, 2)),
                      True, a.quad_logr,
                      0, 0, 0, 0)
    import ipdb; ipdb.set_trace()
    np.testing.assert_almost_equal(G_00[1, 1],
        (14 - 4 * np.log(2) - np.log(16)) / (24 * np.pi))
    # G_00_sing = a.double_integral(a.kernel.displacement_singular, np.zeros((2, 2)),
    #                   True, a.quad_logr,
    #                   0, 0, 0, 0)
    # G_00_nonsing = a.double_integral(a.kernel.displacement_nonsingular, np.zeros((2, 2)),
    #                   True, a.quad_logr,
    #                   0, 0, 0, 0)
    # import ipdb; ipdb.set_trace()
    # G_00_exact = np.array([[7 / (24 * np.pi) - (1 / (18 * np.pi)), 0],
    #                        [0, 7 / (24 * np.pi)]])
    # np.testing.assert_almost_equal(G_00, G_00_exact, 3)

    G_11 = a.double_integral(a.kernel.displacement_kernel, np.zeros((2, 2)),
                      True, a.quad_logr,
                      0, 1, 0, 1)
    np.testing.assert_almost_equal(G_11, np.zeros((2, 2)), 3)

def test_realistic_nan():
    a = realistic_assembler()
    H, G = a.assemble()
    assert(not np.isnan(np.sum(H)))
    assert(not np.isnan(np.sum(G)))

def test_realistic_symmetric_linear():
    a = realistic_assembler()
    H, G = a.assemble()
    # np.testing.assert_almost_equal((H - H.T) / np.mean(H), np.zeros_like(H))
    np.testing.assert_almost_equal((G - G.T) / np.mean(G), np.zeros_like(G))

def test_realistic_double_integral_symmetry():
    a = realistic_assembler(n_elements = 2, element_deg = 1)
    # fnc = lambda r, n: 1 / r[0]
    fnc = a.kernel.displacement_singular
    one = a.double_integral(fnc,
                      np.zeros((2,2)),
                      True,
                      a.quad_oneoverr,
                      1, 0, 1, 1)

    two = a.double_integral(fnc,
                      np.zeros((2,2)),
                      True,
                      a.quad_oneoverr,
                      1, 1, 1, 0)
    np.testing.assert_almost_equal(one, two)

def test_realistic_symmetric_quadratic():
    a = realistic_assembler(n_elements = 1, element_deg = 2)
    H, G = a.assemble()
    # print H-H.T
    # np.testing.assert_almost_equal((H - H.T) / np.mean(H), np.zeros_like(H))
    # Only G should be symmetric for the strongly singular u-BIE.
    np.testing.assert_almost_equal((G - G.T) / np.mean(G), np.zeros_like(G))

def test_reciprocal_effects():
    a = realistic_assembler(n_elements = 2)
    H, G = a.assemble()
    # The influence of u_x(0) on u_y(1) should be the opposite of the
    # effect of u_x(1) on u_y(0), where the parenthesis indicate which element
    np.testing.assert_almost_equal(H[4,0], -H[3,1])
    # The influence of u_y(0) on u_x(1) should be the opposite of the
    # effect of u_y(1) on u_x(0), where the parenthesis indicate which element
    # Really just the symmetric part of the above...
    np.testing.assert_almost_equal(H[5,1], -H[4,2])
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
