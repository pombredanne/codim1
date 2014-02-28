import numpy as np

class Mesh(object):
    """
    A class for managing a one dimensional mesh within a two dimensional
    boundary element code.
    """
    @classmethod
    def simple_line_mesh(cls, n_elements):
        """
        Create a mesh consisting of a line of elements starting at -1 and extending to
        +1 in x coordinate, y = 0.
        """
        n_vertices = n_elements + 1
        vertices = np.zeros((n_vertices, 2))
        x_vals = np.linspace(-1.0, 1.0, n_vertices)
        vertices[:, 0] = x_vals

        element_to_vertex = np.zeros((n_elements, 2))
        for i in range(0, n_elements):
            element_to_vertex[i, :] = (i, i + 1)

        return cls(vertices, element_to_vertex)

    def __init__(self, vertices, element_to_vertex):
        # vertices contains the position of each vertex in tuple form (x, y)
        self.vertices = vertices
        # element_to_vertex contains pairs of indices referring the (x, y) values in
        # vertices
        self.element_to_vertex = element_to_vertex

    def get_physical_points(self, element_id, reference_pts):
        """
        Use a linear affine mapping to convert from the reference element
        back to physical coordinates. Note that the reference element is
        1D whereas physical space is 2D. So, the reference_pts input will be
        scalar and the output will be a 2 element vector.
        """
        vertex_list = self.element_to_vertex[element_id, :]
        pt1 = self.vertices[vertex_list[0]]
        pt2 = self.vertices[vertex_list[1]]
        pt2_minus_pt1 = pt2 - pt1
        physical_pts = pt1 + np.outer(0.5 * (1 + reference_pts), pt2_minus_pt1)
        return physical_pts

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
        return 0.5 * j

################################################################################
# TESTS                                                                        #
################################################################################


def test_simple_line_mesh():
    m = Mesh.simple_line_mesh(2)
    correct_vertices = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    correct_etov = np.array([[0, 1], [1, 2]])
    assert((m.vertices == correct_vertices).all())
    assert((m.element_to_vertex == correct_etov).all())

def test_get_phys_pts():
    m = Mesh.simple_line_mesh(4)

    # Element 2 should lie from 0 to 0.5
    pts = m.get_physical_points(2, np.array([-1.0, 0.0, 1.0]))
    np.testing.assert_almost_equal(pts[0][0], 0.0)
    np.testing.assert_almost_equal(pts[1][0], 0.25)
    np.testing.assert_almost_equal(pts[2][0], 0.5)

def test_jacobian():
    m = Mesh.simple_line_mesh(4)
    j = m.get_element_jacobian(1)
    np.testing.assert_almost_equal(j, 0.25)

