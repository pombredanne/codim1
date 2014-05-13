import numpy as np
from basis_funcs import BasisFunctions
from codim1.fast_lib import MeshEval
from element import Vertex, Element

class Mesh(object):
    """
    A class for managing a one dimensional mesh within a two dimensional
    boundary element code.

    A mesh is defined by vertices and the element that connect them.
    To initialize a mesh, pass a list of vertices and a list of elements.

    Normal vectors are computed assuming that they point left of the vertex
    direction. Meaning, if \vec{r} = \vec{x}_1 - \vec{x}_0 and
    \vec{r} = (r_x, r_y) then \vec{n} = \frac{(r_x, -r_y)}{\norm{r}}. So,
    exterior domain boundaries should be traversed clockwise and
    interior domain boundaries should be traversed counterclockwise.
    """
    def __init__(self, vertices, elements):

        self.basis_fncs = BasisFunctions.from_degree(1)

        # Elements connect vertices and define edge properties
        self.elements = elements
        self.n_elements = len(elements)

        # Vertices contains the position of each vertex in tuple form (x, y)
        self.vertices = vertices
        self.n_vertices = len(self.vertices)

        for e_idx in range(self.n_elements):
            self.elements[e_idx].set_id(e_idx)

        self.compute_connectivity()

        # Compute the coefficients of the mesh basis.
        self.compute_coefficients()

        # The interface with the fast c++ evaluation code.
        self.mesh_eval = MeshEval(self.basis_fncs.fncs,
                                  self.basis_fncs.derivs,
                                  self.coefficients)

    def compute_connectivity(self):
        """Loop over elements and update neighbors."""
        # Determine which elements touch.
        for e in self.elements:
            e.update_neighbors()

    def condense_duplicate_vertices(self, epsilon = 1e-6):
        """
        Remove duplicate vertices and ensure continuity between their
        respective elements. This is a post-processing function on the
        already produced mesh. I don't think it will work in the
        nonlinear mesh case.

        I think it also only condenses pairs of vertices. If three vertices
        all share the same point, the problem is slightly more difficult,
        though the code should be easily adaptable.
        """
        pairs = self._find_equivalent_pairs(epsilon)
        for idx in range(len(pairs)):
            v0 = pairs[idx][0]
            v1 = pairs[idx][1]
            elements_touching = v1.connected_to
            for element in elements_touching:
                if v1 is element.vertex1:
                    element.reinit(v0, element.vertex2)
                if v1 is element.vertex2:
                    element.reinit(element.vertex2, v0)
        self.compute_connectivity()

    def _find_equivalent_pairs(self, epsilon):
        """
        To find equivalent vertex pairs, we sort the list of vertices
        by their x value first and then compare locally within the list.
        """
        # TODO: Should be extendable to find equivalence sets,
        # rather than just pairs.
        sorted_vertices = sorted(self.vertices, key = lambda v: v.loc[0])
        equivalent_pairs = []
        for (idx, v) in enumerate(sorted_vertices[:-1]):
            if np.abs(v.loc[0] - sorted_vertices[idx + 1].loc[0]) > epsilon:
                continue
            if np.abs(v.loc[1] - sorted_vertices[idx + 1].loc[1]) > epsilon:
                continue
            equivalent_pairs.append((v, sorted_vertices[idx + 1]))
        return equivalent_pairs

    def is_neighbor(self, k, l, direction = 'both'):
        """
        Return whether elements k and l are neighbors in the direction
        specified by "direction"
        For example, if element l is to the right of element k, and
        direction == 'right', this method will return true.
        """
        # Check the right side
        if (direction is 'left' or direction is 'both') \
            and self.elements[l] in self.elements[k].neighbors_left:
            return True
        # Check the left side
        if (direction is 'right' or direction is 'both') \
            and self.elements[l] in self.elements[k].neighbors_right:
            return True
        return False

    def get_neighbors(self, k, direction):
        if direction is 'left':
            return self.elements[k].neighbors_left
        if direction is 'right':
            return self.elements[k].neighbors_right
        raise Exception('When calling get_neighbors, direction should be '
                        '\'left\' or \'right\'')

    def compute_coefficients(self):
        # This is basically an interpolation of the boundary function
        # onto the basis
        coefficients = np.empty((2, self.n_elements,
                                 self.basis_fncs.num_fncs))
        for k in range(self.n_elements):
            left_vertex = self.elements[k].vertex1
            right_vertex = self.elements[k].vertex2
            coefficients[:, k, 0] = left_vertex.loc
            coefficients[:, k, 1] = right_vertex.loc
        self.coefficients = coefficients


    def get_physical_point(self, element_idx, x_hat):
        """
        Use the mapping defined by the coefficients and basis functions
        to convert coordinates
        """
        return np.array(self.mesh_eval.get_physical_point(element_idx, x_hat))

    def get_jacobian(self, element_idx, x_hat):
        """
        Use the derivative of the mapping defined by the coefficients/basis
        to get the determinant of the jacobian! This is used to change
        integration coordinates from physical to reference elements.
        """
        return self.mesh_eval.get_jacobian(element_idx, x_hat)

    def get_normal(self, element_idx, x_hat):
        """
        Use the derivative of the mapping to determine the tangent vector
        and thus to determine the local normal vector.
        """
        return np.array(self.mesh_eval.get_normal(element_idx, x_hat))

    def in_element(self, element_idx, point):
        """
        Returns whether the point is within the element specified
        and the reference location of the point if it is within the
        element.
        The probably suboptimal method for a quadratic mesh:
        The mapping from reference coordinates to physical coordinates
        is:
        x = CB\vec{\hat{x}}
        where \vec{\hat{x}} is the reference coordinate to each relevant power
        [\hat{x}^2, \hat{x}^1, 1.0]
        B is a 3x3 matrix representing the basis function and
        C is a 2x3 matrix containing the coefficients.
        If we solve
        \vec{\hat{x}} = (CB)^{-1}x
        then the reference coordinate vector must be consistent for the
        point to lie on the curve. And, for the point to be within the element,
        \hat{x} must be within [0, 1].
        This should work with some minor modifications for higher order
        elements.
        """
        x_coeffs = self.coefficients[0, element_idx, :]
        y_coeffs = self.coefficients[1, element_idx, :]
        basis_vals = self.basis_fncs.fncs
        coeffs_matrix = np.vstack((x_coeffs, y_coeffs))
        mapping_matrix = coeffs_matrix.dot(basis_vals)
        x_hat_row_mapping_matrix = mapping_matrix[:, -2]
        offset = mapping_matrix[:, -1]

        old_settings = np.seterr(divide='ignore', invalid='ignore')
        inv_x_hat_row_mapping_matrix = 1.0 / x_hat_row_mapping_matrix
        x_hat = inv_x_hat_row_mapping_matrix * (point - offset)
        np.seterr(**old_settings)

        on_line = True
        line_pt = 0.0
        if x_hat[1] == x_hat[0]:
            line_pt = x_hat[0]
        elif np.isnan(x_hat[0]):
            line_pt = x_hat[1]
        elif np.isnan(x_hat[1]):
            line_pt = x_hat[0]
        else:
            on_line = False

        if on_line:
            on_segment = (line_pt >= 0.0) and (line_pt <= 1.0)
        else:
            on_segment = False

        return on_segment, line_pt
