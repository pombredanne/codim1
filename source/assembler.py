import quadrature
import numpy as np

class Assembler(object):
    """
    This class builds the matrices needed for a boundary element method.
    Two matrices are created. The terminology here is in reference to the
    derivation based on the reciprocal theorem. The definitions of these
    matrices can be derived by performing integration by parts on the
    original governing equations of static elasticity.

    1. G -- The traction kernel matrix represents the test tractions acting
    through the solution displacements at each node of each element.

    2. H -- The displacement kernel matrix represents the test displacements
    acting through the solution tractions at each node of each element.
    """
    def __init__(self,
                 mesh,
                 basis_funcs,
                 kernel,
                 dof_map,
                 quad_points_nonsingular,
                 quad_points_logr,
                 quad_points_oneoverr):
        self.mesh = mesh
        self.basis_funcs = basis_funcs
        self.kernel = kernel
        self.dof_map = dof_map
        self.quad_points_nonsingular = quad_points_nonsingular
        self.quad_points_logr = quad_points_logr
        self.quad_points_oneoverr = quad_points_oneoverr

        self.setup_quadrature()

    def setup_quadrature(self):
        """
        The quadrature rules can be defined once on the reference element
        and then a change of variables allows integration on any element.

        We need (element_deg) * (element_deg + 1) different quadrature
        rules for the singular case. Two for each pair of reference element distances.
        One for the 1/(log r) case and one for the 1/r case.
        """
        self.quad_nonsingular = quadrature.QuadGauss(self.quad_points_nonsingular)
        self.quad_logr = []
        self.quad_oneoverr = []
        for i in range(self.basis_funcs.num_fncs):
            inside_logr = [0] * self.basis_funcs.num_fncs
            inside_oneoverr = [0] * self.basis_funcs.num_fncs
            # The set of distances is symmetric so only loop from
            # i to n_fncs
            for j in range(i, self.basis_funcs.num_fncs):
                x1 = self.basis_funcs.nodes[i]
                x2 = self.basis_funcs.nodes[j]
                r = x2 - x1
                logr = quadrature.QuadGaussLogR(self.quad_points_logr, 1.0, r)
                oneoverr = quadrature.QuadGaussOneOverR(
                                                        self.quad_points_oneoverr, r)
                inside_logr[j] = logr
                inside_oneoverr[j] = oneoverr
            self.quad_logr.append(inside_logr)
            self.quad_oneoverr.append(inside_oneoverr)
        # Make the set of quadrature formulas symmetric
        for i in range(self.basis_funcs.num_fncs):
            for j in range(i):
                self.quad_logr[i][j] = self.quad_logr[j][i]
                self.quad_oneoverr[i][j] = self.quad_oneoverr[j][i]

    def get_quadrature(self, GHM, k, l, i, j):
        if k != l or GHM == 'nonsingular':
            x = self.quad_nonsingular.x
            w = self.quad_nonsingular.w
            return (x, w)
        if GHM == 'oneoverr':
            x = self.quad_oneoverr[i][j].x
            w = self.quad_oneoverr[i][j].w
        elif GHM == 'logr':
            x = self.quad_logr[i][j].x
            w = self.quad_logr[i][j].w
        return (x, w)

    def assemble():
        total_dofs = 2 * self.mesh.n_elements * self.basis_funcs.num_fncs
        G = np.empty(total_dofs, total_dofs)
        H = np.empty(total_dofs, total_dofs)
        for k in range(self.mesh.n_elements):
            for i in range(self.basis_funcs.num_fncs):
                assemble_G_row(k, i)
                assemble_H_row(k, i)

    def assemble_H_row(self, k, i):
        """
        Assemble one row of the H matrix.
        """
        H_row_x = np.zeros(2 * self.mesh.n_elements * self.basis_funcs.num_fncs)
        H_row_y = np.zeros(2 * self.mesh.n_elements * self.basis_funcs.num_fncs)
        for l in range(self.mesh.n_elements):
            for j in range(self.basis_funcs.num_fncs):
                src_dof_x = self.dof_map[0, l, j]
                src_dof_y = self.dof_map[1, l, j]

                H_local = self.assemble_H_one_interaction(k, i, l, j)
                # Add the local interactions to the global matrix in
                # the proper locations
                H_row_x[src_dof_x] += H_local[0, 0]
                H_row_x[src_dof_y] += H_local[0, 1]
                H_row_y[src_dof_x] += H_local[1, 0]
                H_row_y[src_dof_y] += H_local[1, 1]
        return H_row_x, H_row_y


    def assemble_G_row(self, k, i):
        """
        Assemble one row of the G matrix.
        """
        G_row_x = np.zeros(2 * self.mesh.n_elements * self.basis_funcs.num_fncs)
        G_row_y = np.zeros(2 * self.mesh.n_elements * self.basis_funcs.num_fncs)
        for l in range(self.mesh.n_elements):
            for j in range(self.basis_funcs.num_fncs):
                src_dof_x = self.dof_map[0, l, j]
                src_dof_y = self.dof_map[1, l, j]

                if k == l:
                    M_local = self.assemble_M_one_interaction(k, i, j)
                    # M_local is only applied on the block diagonal
                    G_row_x[src_dof_x] += M_local
                    G_row_y[src_dof_y] += M_local

                G_local = self.assemble_G_one_interaction(k, i, l, j)
                # Add the local interactions to the global matrix in
                # the proper locations
                G_row_x[src_dof_x] += G_local[0, 0]
                G_row_x[src_dof_y] += G_local[0, 1]
                G_row_y[src_dof_x] += G_local[1, 0]
                G_row_y[src_dof_y] += G_local[1, 1]
        return G_row_x, G_row_y

    def assemble_H_one_interaction(self, k, i, l, j):
        """
        Compute one pair of element interactions for the H matrix.
        """
        q_pts, w = self.get_quadrature('logr', k, l, i, j)
        H_sing = np.zeros((2, 2))
        import ipdb; ipdb.set_trace()
        H_sing = self.double_integral(self.kernel.displacement_singular,
                             H_sing,
                             q_pts,
                             w,
                             k, i, l, j)

        q_pts, w = self.get_quadrature('nonsingular', k, l, i, j)
        H_nonsing = np.zeros((2, 2))
        H_nonsing = self.double_integral(self.kernel.displacement_nonsingular,
                             H_nonsing,
                             q_pts,
                             w,
                             k, i, l, j)
        return H_sing + H_nonsing

    def assemble_M_one_interaction(self, k, i, j):
        """
        Compute one local mass matrix interaction.
        """
        q_pts, w = self.get_quadrature('nonsingular', k, k, i, j)
        M_local = self.single_integral(lambda x: 1.0, 0.0, q_pts, w,
                             k, i, j)
        return -0.5 * M_local

    def assemble_G_one_interaction(self, k, i, l, j):
        """
        Compute one pair of element interactions for the G matrix.
        """
        q_pts, w = self.get_quadrature('oneoverr', k, l, i, j)
        G_local = np.zeros((2, 2))
        G_local = self.double_integral(self.kernel.traction_kernel,
                             G_local,
                             q_pts,
                             w,
                             k, i, l, j)
        return G_local

    def single_integral(self, kernel, result, q_pts, w, k, i, j):
        """
        Performs a single integral over the element specified by k
        with the basis functions specified by i and j. q_pts and w
        define the quadrature rule. kernel should be a function that
        can be evaluated at all point within the element
        """
        jacobian = self.mesh.get_element_jacobian(k)
        for (q_pt, w) in zip(q_pts, w):
            # The basis functions should be evaluated on reference
            # coordinates
            soln_basis_fnc = self.basis_funcs.evaluate_basis(i, q_pt)
            src_basis_fnc = self.basis_funcs.evaluate_basis(j, q_pt)
            phys_pt = self.mesh.get_physical_points(k, q_pt)[0]
            result += kernel(phys_pt) * \
                soln_basis_fnc * src_basis_fnc * jacobian * w
        return result

    def double_integral(self, kernel, result, q_pts, w, k, i, l, j):
        """
        Performs a double integral over a pair of elements with the
        provided quadrature rule.

        Warning: This function modifies the "result" input.
        """
        soln_jacobian = self.mesh.get_element_jacobian(k)
        src_jacobian = self.mesh.get_element_jacobian(l)
        # The normal is the one on the soln integration element, because
        # this is the
        normal = self.mesh.normals[k]
        for (q_pt_soln, w_soln) in zip(q_pts[0], w[0]):
            phys_soln_pt = self.mesh.get_physical_points(k, q_pt_soln)[0]
            # The basis functions should be evaluated on reference
            # coordinates
            soln_basis_fnc = self.basis_funcs.evaluate_basis(i, q_pt_soln)
            for (q_pt_src, w_src) in zip(q_pts[1], w[1]):
                src_basis_fnc = self.basis_funcs.evaluate_basis(j, q_pt_src)

                # Separation of the two quadrature points, use real,
                # physical coordinates!
                phys_src_pt = self.mesh.get_physical_points(l, q_pt_src)[0]
                r = phys_soln_pt - phys_src_pt

                T = kernel(r, normal)

                # Actually perform the quadrature
                result += T * src_basis_fnc * soln_basis_fnc *\
                              soln_jacobian * src_jacobian *\
                              w_soln * w_src
        return result



################################################################################
# TESTS                                                                        #
################################################################################

import basis_funcs
import elastic_kernel
import mesh

class TestKernel(object):
    """
    This class exists to assist with testing matrix assembly.
    The normal kernels are too complex to make testing easy.
    """

    def displacement_singular(self, r, n):
        dist = np.sqrt(r[0] ** 2 + r[1] ** 2)
        return np.array([[np.log(dist), 0.0], [0.0, np.log(dist)]])

    def displacement_nonsingular(self, r, n):
        return np.ones((2, 2))

    def displacement_kernel(self, r, n):
        return np.ones((2, 2))

    def traction_kernel(self, r, n):
        return np.ones((2, 2))

def simple_assembler(degree = 0,
                     nonsing_pts = 1
                     logr_pts = 2,
                     oneoverr_pts = 2,
                     n_elements = 2):
    dof_map = np.arange(2 * n_elements * (degree + 1)).\
              reshape(2, n_elements, degree + 1)
    bf = basis_funcs.BasisFunctions.from_degree(degree)
    k = TestKernel()
    msh = mesh.Mesh.simple_line_mesh(n_elements)
    assembler = Assembler(msh, bf, k, dof_map,
                          nonsing_pts, logr_pts, oneoverr_pts)
    return assembler

def test_build_quadrature_list():
    a = simple_assembler(degree = 2)

    assert(a.quad_nonsingular.N == 1)

    assert(len(a.quad_logr) == 3)
    assert(len(a.quad_oneoverr) == 3)
    assert(len(a.quad_logr[0]) == 3)
    assert(len(a.quad_oneoverr[0]) == 3)

    assert(a.quad_logr[0][0].N == 2)
    assert(a.quad_oneoverr[0][0].N == 2)

    assert(a.quad_logr[0][0].x0 == 0)
    assert(a.quad_logr[1][0].x0 == 0.5)
    assert(a.quad_logr[2][0].x0 == 1.0)
    assert(a.quad_oneoverr[0][0].x0 == 0)
    assert(a.quad_oneoverr[1][0].x0 == 0.5)
    assert(a.quad_oneoverr[2][0].x0 == 1.0)

def test_get_quadrature():
    a = simple_assembler(degree = 2)
    x1, w = a.get_quadrature('oneoverr', 1, 1, 1, 2)
    assert(len(x1) == 6)

    x2, w = a.get_quadrature('logr', 0, 0, 0, 1)
    assert(len(x2) == 8)

    x3, w = a.get_quadrature('nonsingular', 1, 1, 2, 2)
    assert(len(x3) == 1)

def test_assemble_G_one_element_off_diagonal():
    a = simple_assembler(oneoverr_pts = 2)
    G_local = a.assemble_G_one_interaction(0, 0, 1, 0)
    assert((G_local == np.ones((2, 2))).all())

def test_assemble_G_one_element_on_diagonal():
    a = simple_assembler(oneoverr_pts = 2)
    G_local = a.assemble_G_one_interaction(0, 0, 0, 0)
    np.testing.assert_almost_equal(G_local, np.ones((2,2)))

def test_assemble_G_row_test_kernel():
    a = simple_assembler(oneoverr_pts = 4)

    # The row functions should return one vector for each dimension.
    G_row_x, G_row_y = a.assemble_G_row(0, 0)

    np.testing.assert_almost_equal(G_row_x, np.array([0.5, 1.0, 1.0, 1.0]))
    np.testing.assert_almost_equal(G_row_y, np.array([1.0, 1.0, 0.5, 1.0]))

def test_assemble_H_one_element_off_diagonal():
    a = simple_assembler(logr_pts = 2)
    H_local = a.assemble_H_one_interaction(0, 0, 1, 0)
    assert((H_local == np.array([[1.0, 1.0], [1.0, 1.0]])).all())

def test_assemble_H_one_element_on_diagonal():
    a = simple_assembler(logr_pts = 2)
    H_local = a.assemble_H_one_interaction(0, 0, 0, 0)
    assert((H_local == np.array([[0.0, 0.0], [0.0, 0.0]])).all())

def test_assemble_H_row_test_kernel():
    a = simple_assembler(logr_pts = 2)

    # The row functions should return one vector for each dimension.
    H_row_x, H_row_y = a.assemble_H_row(0, 0)

    assert((H_row_x == np.array([-0.5, 1.0, 0.0, 1.0])).all())
    assert((H_row_y == np.array([0, 1.0, -0.5, 1.0])).all())




