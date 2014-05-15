import numpy as np
from segment_distance import segments_distance
from basis_funcs import BasisFunctions
from codim1.fast_lib import MappingEval

def apply_mapping(mesh, mapping_gen):
    """ Apply the same mapping to all elements in a mesh. """
    for e in mesh.elements:
        e.mapping = mapping_gen(e)

def distance_between_mappings(el1, el2):
    """
    A coarsely accurate computation of the distance between two
    elements. The distance is computed by finding the minimum distance
    between the segments of a linear approximation to the element.
    """
    el1_verts = el1.get_linear_approximation()
    el2_verts = el2.get_linear_approximation()
    min_dist = 1e50
    for idx1 in range(len(el1_verts) - 1):
        for idx2 in range(len(el2_verts) - 1):
            dist = segments_distance(el1_verts[idx1].loc[0],
                                     el1_verts[idx1].loc[1],
                                     el1_verts[idx1 + 1].loc[0],
                                     el1_verts[idx1 + 1].loc[1],
                                     el2_verts[idx2].loc[0],
                                     el2_verts[idx2].loc[1],
                                     el2_verts[idx2 + 1].loc[0],
                                     el2_verts[idx2 + 1].loc[1])
            min_dist = min(dist, min_dist)
    return min_dist


class PolynomialMapping(object):
    """
    This class manages the mapping between physical coordinates and reference
    coordinates. Most of the time, this class will be used as a linear
    mapping. However, if a curved mesh is desired, a higher degree mapping
    can be used. Just specify degree > 1 and a boundary function. It
    is also necessary to properly specify the vertex parameter for each
    vertex. These will be linearly interpolated when a new point is added.

    Besides translating reference to physical points, the class also
    provides jacobians to be used when a change of variables is performed
    under an integral. Normal vectors are also provided.
    """
    def __init__(self, element, degree = 1, boundary_function = None):
        if degree > 1 and boundary_function is None:
            raise Exception("Sorry. A boundary function is needed in" +
                            " order to describe a higher order polynomial" +
                            " mapping. Either choose degree = 1 or provide" +
                            " a boundary function.")

        self.element = element
        self.basis_fncs = BasisFunctions.from_degree(degree)
        self.boundary_function = boundary_function

        # Compute the coefficients of the mapping basis.
        self.compute_coefficients()

        # The interface with the fast c++ evaluation code.
        self.eval = MappingEval(self.basis_fncs.fncs,
                                  self.basis_fncs.derivs,
                                  self.coefficients)

    def compute_coefficients(self):
        # This is basically an interpolation of the boundary function
        # onto the basis
        self.coefficients = np.empty((2, self.basis_fncs.num_fncs))
        left_vertex = self.element.vertex1
        right_vertex = self.element.vertex2
        left_param = left_vertex.param
        right_param = right_vertex.param
        self.coefficients[:, 0] = left_vertex.loc
        for i in range(1, self.basis_fncs.num_fncs - 1):
            x_hat = self.basis_fncs.nodes[i]
            t = (1 - x_hat) * left_param + x_hat * right_param
            self.coefficients[:, i] = self.boundary_function(t)
        self.coefficients[:, -1] = right_vertex.loc

    def get_physical_point(self, x_hat):
        """
        Use the mapping defined by the coefficients and basis functions
        to convert coordinates
        """
        return np.array(self.eval.get_physical_point(x_hat))

    def get_jacobian(self, x_hat):
        """
        Use the derivative of the mapping defined by the coefficients/basis
        to get the determinant of the jacobian! This is used to change
        integration coordinates from physical to reference elements.
        """
        return self.eval.get_jacobian(x_hat)

    def get_normal(self, x_hat):
        """
        Use the derivative of the mapping to determine the tangent vector
        and thus to determine the local normal vector.
        """
        return np.array(self.eval.get_normal(x_hat))

    def in_element(self, point):
        """
        Returns whether the point is within the element specified
        and the reference location of the point if it is within the
        element.
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
        """
        x_coeffs = self.coefficients[0, :]
        y_coeffs = self.coefficients[1, :]
        basis_vals = self.basis_fncs.fncs
        mapping_matrix = self.coefficients.dot(basis_vals)
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

    def get_linear_approximation(self):
        """
        Computes a small number of linear segments that approximate this
        element.
        """
        verts = [self.element.vertex1]
        for i in range(1, self.basis_fncs.num_fncs - 1):
            x_hat = self.basis_fncs.nodes[i]
            phys_pt = self.get_physical_point(x_hat)
            verts.append(Vertex(phys_pt))
        verts.append(self.element.vertex2)
        return verts
