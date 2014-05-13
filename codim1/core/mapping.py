import numpy as np
from basis_funcs import BasisFunctions
from codim1.fast_lib import MappingEval

def apply_mapping(mesh, mapping_type):
    """ Apply the same mapping to all elements in a mesh. """
    for e in mesh.elements:
        e.mapping = mapping_type(e)


class LinearMapping(object):
    """
    This class manages the mapping between physical coordinates and reference
    coordinates. In this case, the mapping is linear. However, the same
    interface could be implemented for a variety of nonlinear mappings. The
    simplest would be higher order polynomial mappings.

    Besides translating reference to physical points, the class also
    provides jacobians to be used when a change of variables is performed
    under an integral. Normal vectors are also provided.
    """
    def __init__(self, element):
        self.element = element
        self.basis_fncs = BasisFunctions.from_degree(1)

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
        self.coefficients[:, 0] = left_vertex.loc
        self.coefficients[:, 1] = right_vertex.loc

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
        x_coeffs = self.coefficients[0, :]
        y_coeffs = self.coefficients[1, :]
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
