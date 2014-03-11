import numpy as np
from codim1.fast.get_physical_points import get_physical_points as _get_physical_points

class Mesh(object):
    """
    A class for managing a one dimensional mesh within a two dimensional
    boundary element code.

    Normal vectors are computed assuming that they point left of the vertex
    direction. Meaning, if \vec{r} = \vec{x}_1 - \vec{x}_0 and
    \vec{r} = (r_x, r_y) then \vec{n} = \frac{(r_x, -r_y)}{\norm{r}}. So,
    exterior domain boundaries should be traversed clockwise and
    interior domain boundaries should be traversed counterclockwise.
    """
    def __init__(self, vertices, element_to_vertex):
        # vertices contains the position of each vertex in tuple form (x, y)
        self.vertices = vertices
        # element_to_vertex contains pairs of indices referring the (x, y)
        # values in vertices
        self.element_to_vertex = element_to_vertex

        self.n_vertices = vertices.shape[0]
        self.n_elements = element_to_vertex.shape[0]

        self.compute_normals()
        self.compute_connectivity()
        # self.compute_mappings()

    @classmethod
    def simple_line_mesh(cls, n_elements, left_edge = -1.0, right_edge = 1.0):
        """
        Create a mesh consisting of a line of elements starting at -1 and
        extending to +1 in x coordinate, y = 0.
        """
        n_vertices = n_elements + 1
        vertices = np.zeros((n_vertices, 2))
        x_vals = np.linspace(left_edge, right_edge, n_vertices)
        vertices[:, 0] = x_vals

        element_to_vertex = np.zeros((n_elements, 2))
        for i in range(0, n_elements):
            element_to_vertex[i, :] = (i, i + 1)
        element_to_vertex = element_to_vertex.astype(int)

        return cls(vertices, element_to_vertex)

    @classmethod
    def circular_mesh(cls, n_elements, radius):
        n_vertices = n_elements
        theta = np.linspace(0, 2 * np.pi, n_vertices + 1)[:-1]
        vertices = np.zeros((n_vertices, 2))
        vertices[:, 0] = radius * np.cos(theta)
        vertices[:, 1] = radius * np.sin(theta)

        element_to_vertex = np.zeros((n_elements, 2))
        for i in range(0, n_elements - 1):
            element_to_vertex[i, :] = (i, i + 1)
        element_to_vertex[-1, :] = (n_elements - 1, 0)
        element_to_vertex = element_to_vertex.astype(int)
        return cls(vertices, element_to_vertex)

    def compute_normals(self):
        self.normals = np.empty((self.n_elements, 2))
        all_vertices = self.vertices[self.element_to_vertex]
        r = (all_vertices[:,1,:] - all_vertices[:,0,:])
        r_norm = np.linalg.norm(r, axis = 1)
        self.normals = np.vstack((-r[:, 1] / r_norm, r[:, 0] / r_norm)).T

    # def compute_mappings(self):
    #     pt1 = self.vertices[self.element_to_vertex[:, 0]]
    #     pt2 = self.vertices[self.element_to_vertex[:, 1]]
    #     self.pt2_minus_pt1 = pt2 - pt1

    def compute_connectivity(self):
        """
        Determine a store a representation of which elements are adjacent.
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
            self.neighbors[k][0] = touch_vertex[self.element_to_vertex[k][0]][0]
            self.neighbors[k][1] = touch_vertex[self.element_to_vertex[k][1]][1]
        self.neighbors = self.neighbors.astype(np.int)

    def is_neighbor(self, k, l, direction = 'both'):
        # Check the right side
        if (direction is 'left' or direction is 'both') \
            and self.neighbors[k][0] == l:
            return True
        # Check the left side
        if (direction is 'right' or direction is 'both') \
            and self.neighbors[k][1] == l:
            return True
        return False

    def get_physical_points(self, element_id, reference_pt):
        """
        Use a linear affine mapping to convert from the reference element
        back to physical coordinates. Note that the reference element is
        1D whereas physical space is 2D. So, the reference_pt input will be
        scalar and the output will be a 2 element vector.
        """
        return _get_physical_points(self.element_to_vertex,
                                    self.vertices,
                                    element_id,
                                    reference_pt)

    def get_element_jacobian(self, element_id):
        """
        Returns the jacobian of the linear affine mapping from the
        reference element to physical space.
        Used for evaluating integrals on the reference element rather than in
        physical coordinates.
        """
        vertex_list = self.element_to_vertex[element_id, :]
        pt1 = self.vertices[vertex_list[0]]
        pt2 = self.vertices[vertex_list[1]]
        pt2_minus_pt1 = pt2 - pt1
        # Take the length of the line segment between the points. This
        # is the relevant jacobian for a line integral change of variables.
        j = np.sqrt(pt2_minus_pt1[0] ** 2 + pt2_minus_pt1[1] ** 2)
        return j
