from codim1.core.quadrature import QuadGauss
from codim1.fast_lib import ConstantBasis, MassMatrixKernel, single_integral

def apply_average_constraint(matrix, rhs, mesh):
    """
    Convert the first two rows of the matrix and right hand side
    into a constraint that enforces a total solution in both the x and y
    direction of 0.
    """
    # Set the rhs to 0 for the first two rows
    first_x_dof = mesh.elements[0].dofs[0, 0]
    rhs[first_x_dof] = 0
    first_y_dof = mesh.elements[0].dofs[1, 0]
    rhs[first_y_dof] = 0

    kernel = MassMatrixKernel(0, 0)
    one = ConstantBasis([1.0, 1.0])
    for e_k in mesh:
        for i in range(e_k.basis.n_fncs):
            # TODO: Either pass in a quad_strategy object or make sure this
            # behaves like a flyweight
            quad_info = QuadGauss(e_k.basis.n_fncs).quad_info
            dof_x = e_k.dofs[0,i]
            dof_y = e_k.dofs[1, i]
            integral = single_integral(e_k.mapping.eval,
                                       kernel,
                                       e_k.basis,
                                       one,
                                       quad_info,
                                       i, 0)
            matrix[first_x_dof, dof_x] += integral[0][0]
            matrix[first_y_dof, dof_y] += integral[1][1]

# TODO: Refactor this into a "pin-element" constraint that takes an
# element id as an input and a left-right flag
def pin_ends_constraint(matrix, rhs, mesh, left_end, right_end):
    """
    Pin the values at the ends of a simple line mesh. This is useful
    for crack solutions with only traction boundary conditions, because
    the matrix is singular. Pinning the ends removes the rigid body motion
    modes of solution and results in a nonsingular matrix.
    """
    # TODO: This is dependent on the linear storage of the elements
    # and would not be compatible with a tree structure like in a FMM
    # BAD!
    first_x_dof = mesh.elements[0].dofs[0, 0]
    last_x_dof = mesh.elements[-1].dofs[0, -1]
    first_y_dof = mesh.elements[0].dofs[1, 0]
    last_y_dof = mesh.elements[-1].dofs[1, -1]
    matrix[first_x_dof, :] = 0.0
    matrix[first_x_dof, first_x_dof] = 1.0
    rhs[first_x_dof] = left_end[0]
    matrix[last_x_dof, :] = 0.0
    matrix[last_x_dof, last_x_dof] = 1.0
    rhs[last_x_dof] = right_end[0]
    matrix[first_y_dof, :] = 0.0
    matrix[first_y_dof, first_y_dof] = 1.0
    rhs[first_y_dof] = left_end[1]
    matrix[last_y_dof, :] = 0.0
    matrix[last_y_dof, last_y_dof] = 1.0
    rhs[last_y_dof] = right_end[1]
