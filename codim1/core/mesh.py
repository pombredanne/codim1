import numpy as np
from segment_distance import segments_distance
from basis_funcs import BasisFunctions
from codim1.fast_lib import MeshEval

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
    def __init__(self, vertices, element_to_vertex):

        self.basis_fncs = BasisFunctions.from_degree(1)

        # element_to_vertex contains pairs of indices referring the (x, y)
        # values in vertices
        self.element_to_vertex = element_to_vertex
        self.n_elements = element_to_vertex.shape[0]

        # Vertices contains the position of each vertex in tuple form (x, y)
        self.vertices = vertices
        self.n_vertices = self.vertices.shape[0]

        # Determine which elements touch.
        self.compute_connectivity()

        # Compute the coefficients of the mesh basis.
        self.compute_coefficients()

        # Compute the separation between elements
        self.compute_element_distances()

        # The length of each element.
        self.compute_element_widths()

        # The evaluation class that operates in the c++ layer.
        self.mesh_eval = MeshEval(self.basis_fncs.fncs,
                                  self.basis_fncs.derivs,
                                  self.coefficients)
        self.parts = []

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
        for idx in range(pairs.shape[0]):
            self.element_to_vertex[self.element_to_vertex == pairs[idx, 1]] =\
                pairs[idx, 0]
        self.compute_connectivity()

    def _find_equivalent_pairs(self, epsilon):
        # TODO: Should be extendable to find equivalence sets,
        # rather than just pairs.
        sorted_vertices = self.vertices[self.vertices[:,0].argsort()]
        equivalent_pairs = []
        for (idx, x_val) in enumerate(sorted_vertices[:-1, 0]):
            if np.abs(x_val - sorted_vertices[idx + 1, 0]) > epsilon:
                continue
            if np.abs(sorted_vertices[idx, 1] - sorted_vertices[idx + 1, 1])\
                > epsilon:
                continue
            equivalent_pairs.append((idx, idx + 1))
        return np.array(equivalent_pairs)

    def compute_coefficients(self):
        # This is basically an interpolation of the boundary function
        # onto the basis
        coefficients = np.empty((2, self.n_elements,
                                 self.basis_fncs.num_fncs))
        for k in range(self.n_elements):
            left_vertex = self.vertices[self.element_to_vertex[k, 0]]
            right_vertex = self.vertices[self.element_to_vertex[k, 1]]
            coefficients[:, k, 0] = left_vertex
            coefficients[:, k, 1] = right_vertex
        self.coefficients = coefficients

    def compute_element_distances(self):
        """
        Compute the pairwise distance between all the elements. In
        2D, this is just the pairwise line segment distances. Moving to 3D,
        this shouldn't be hard if the polygons are "reasonable", but handling
        outliers may be harder. Because this distance is only used for
        selecting the quadrature strategy, I should be conservative. Using too
        many quadrature points is not as bad as using too few. Using a
        bounding box method might be highly effective.
        """
        self.element_distances = np.zeros((self.n_elements, self.n_elements))
        for k in range(self.n_elements):
            o1 = self.element_to_vertex[k, 0]
            o2 = self.element_to_vertex[k, 1]
            outer_vertex1 = self.vertices[o1, :]
            outer_vertex2 = self.vertices[o2, :]
            # Only loop over the upper triangle of the matrix
            for l in range(k, self.n_elements):
                i1 = self.element_to_vertex[l, 0]
                i2 = self.element_to_vertex[l, 1]
                inner_vertex1 = self.vertices[i1, :]
                inner_vertex2 = self.vertices[i2, :]
                dist = segments_distance(outer_vertex1[0], outer_vertex1[1],
                                         outer_vertex2[0], outer_vertex2[1],
                                         inner_vertex1[0], inner_vertex1[1],
                                         inner_vertex2[0], inner_vertex2[1])
                self.element_distances[k, l] = dist
        # Make it symmetric. No need to worry about doubling the diagonal
        # because the diagonal *should* be zero!
        self.element_distances += self.element_distances.T

    def compute_element_widths(self):
        """
        Calculate the length of each element.
        """
        self.element_widths = np.empty(self.n_elements)
        for k in range(self.n_elements):
            v1 = self.vertices[self.element_to_vertex[k, 0], :]
            v2 = self.vertices[self.element_to_vertex[k, 1], :]
            length = np.sqrt((v2[1] - v1[1]) ** 2 + (v2[0] - v1[0]) ** 2)
            self.element_widths[k] = length

    def compute_connectivity(self):
        """
        Determine and store a representation of which elements are adjacent.
        Simple for 2D.
        """
        # Create of list of which elements touch each vertex
        # -1 if no element touches there
        touch_vertex = -np.ones((self.n_vertices, 2))
        for k in range(0, self.n_elements):
            # The left vertex of an element has that element
            # as its right neighbor and vice-versa
            touch_vertex[self.element_to_vertex[k][0]][1] = k
            touch_vertex[self.element_to_vertex[k][1]][0] = k

        self.neighbors = np.zeros((self.n_elements, 2))
        for k in range(0, self.n_elements):
            self.neighbors[k][0] = \
                touch_vertex[self.element_to_vertex[k][0]][0]
            self.neighbors[k][1] = \
                touch_vertex[self.element_to_vertex[k][1]][1]
        self.neighbors = self.neighbors.astype(np.int)

    def is_neighbor(self, k, l, direction = 'both'):
        """
        Return whether elements k and l are neighbors in the direction
        specified by "direction"
        For example, if element l is to the right of element k, and
        direction == 'right', this method will return true.
        """
        # Check the right side
        if (direction is 'left' or direction is 'both') \
            and self.neighbors[k][0] == l:
            return True
        # Check the left side
        if (direction is 'right' or direction is 'both') \
            and self.neighbors[k][1] == l:
            return True
        return False

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
