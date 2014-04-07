import numpy as np
from segment_distance import segments_distance
from basis_funcs import BasisFunctions
from codim1.fast_lib import MeshEval

class Mesh(object):
    """
    A class for managing a one dimensional mesh within a two dimensional
    boundary element code.

    This mesh class can use higher order mappings.
    Arbitrary order of mapping can be used by simply providing the proper
    set of basis functions. This is unnecessary for problems with
    straight boundaries.

    Some of the functions that are simple for a linear boundary are much more
    difficult for higher order boundaries. For example, computing the distance
    between two elements is more involved.
    Also, for a linear element, the jacobian determinant and the normal
    vectors are constant. The calling code should account for this.

    Normal vectors are computed assuming that they point left of the vertex
    direction. Meaning, if \vec{r} = \vec{x}_1 - \vec{x}_0 and
    \vec{r} = (r_x, r_y) then \vec{n} = \frac{(r_x, -r_y)}{\norm{r}}. So,
    exterior domain boundaries should be traversed clockwise and
    interior domain boundaries should be traversed counterclockwise.

    boundary_fnc is a function that defines the boundary.
    """
    def __init__(self, boundary_fnc,
                       vertex_params, element_to_vertex,
                       basis_fncs = None,
                       element_to_vertex_params = None):

        # Default to linear mesh.
        self.basis_fncs = basis_fncs
        if self.basis_fncs is None:
            self.basis_fncs = BasisFunctions.from_degree(1)

        # element_to_vertex contains pairs of indices referring the (x, y)
        # values in vertices
        self.element_to_vertex = element_to_vertex
        self.n_elements = element_to_vertex.shape[0]

        # A linear mesh basis should be treated specially
        if self.basis_fncs.num_fncs > 2:
            self.is_linear = False
        else:
            self.is_linear = True
        if self.basis_fncs.num_fncs < 2:
            raise Exception("At least two basis functions are required to" +
                            " describe a mesh edge.")

        # Vertices contains the position of each vertex in tuple form (x, y)
        self.vertex_params = vertex_params
        self.boundary_fnc = boundary_fnc
        self.n_vertices = self.vertex_params.shape[0]
        self.vertices = np.empty((self.n_vertices, 2))
        for (i, vp) in enumerate(vertex_params):
            self.vertices[i, :] = boundary_fnc(vp)

        # Default is that the parameters and vertices match up.
        # See the circular mesh example for where this constraint is broken.
        self.element_to_vertex_params = element_to_vertex_params
        if self.element_to_vertex_params is None:
            self.element_to_vertex_params = \
                    self.vertex_params[element_to_vertex]

        # Determine which elements touch.
        self.compute_connectivity()

        # Compute the coefficients of the mesh basis.
        self.compute_coefficients()

        self.compute_element_distances()
        self.compute_element_widths()

        self.mesh_eval = MeshEval(self.basis_fncs.fncs,
                                  self.basis_fncs.derivs,
                                  self.coefficients)

    @classmethod
    def simple_line_mesh(cls, n_elements,
            left_edge = (-1.0, 0.0), right_edge = (1.0, 0.0)):
        """
        Create a mesh consisting of a line of elements starting at -1 and
        extending to +1 in x coordinate, y = 0.
        """
        if type(left_edge) is float:
            left_edge = (left_edge, 0.0)
        if type(right_edge) is float:
            right_edge = (right_edge, 0.0)
        n_vertices = n_elements + 1
        vertex_params = np.linspace(left_edge[0], right_edge[0], n_vertices)
        #slope
        m = (right_edge[1] - left_edge[1]) / (right_edge[0] - left_edge[0])
        c = right_edge[0]
        d = right_edge[1]
        if abs(m) < 0.000001:
            vertex_function = lambda x: np.array([x, d])
        else:
            vertex_function = lambda x: np.array([x, c - ((d - x) / m)])

        element_to_vertex = np.zeros((n_elements, 2))
        for i in range(0, n_elements):
            element_to_vertex[i, :] = (i, i + 1)
        element_to_vertex = element_to_vertex.astype(int)

        return cls(vertex_function, vertex_params, element_to_vertex)

    @classmethod
    def circular_mesh(cls, n_elements, radius, basis_fncs = None):
        # Use linear basis by default
        if basis_fncs is None:
            basis_fncs = BasisFunctions.from_degree(1)

        n_vertices = n_elements
        vertex_params = np.linspace(0, 2 * np.pi, n_vertices + 1)
        boundary_func = lambda t: radius * np.sin([np.pi / 2 - t, t])

        element_to_vertex = np.zeros((n_elements, 2))
        element_to_vertex_params = np.zeros((n_elements, 2))
        for i in range(0, n_elements):
            element_to_vertex[i, :] = (i, i + 1)
            element_to_vertex_params[i, :] = (vertex_params[i],
                                              vertex_params[i + 1])
        element_to_vertex[-1, :] = (n_elements - 1, 0)
        element_to_vertex = element_to_vertex.astype(int)

        C = cls(boundary_func,
                vertex_params[:-1], element_to_vertex,
                basis_fncs, element_to_vertex_params)
        return C

    def compute_coefficients(self):
        # This is basically an interpolation of the boundary function
        # onto the basis
        coefficients = np.empty((2, self.n_elements,
                                 self.basis_fncs.num_fncs))
        for k in range(self.n_elements):
            left_param = self.element_to_vertex_params[k, 0]
            right_param = self.element_to_vertex_params[k, 1]
            for (i, node) in enumerate(self.basis_fncs.nodes):
                node_param = left_param + node * (right_param - left_param)
                node_loc = self.boundary_fnc(node_param)
                coefficients[:, k, i] = node_loc
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
        return self.mesh_eval.get_physical_point(element_idx, x_hat)

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
        return self.mesh_eval.get_normal(element_idx, x_hat)
