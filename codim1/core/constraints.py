from quadrature import QuadGauss
from codim1.fast_lib import ConstantEval, MassMatrixKernel, single_integral

def apply_average_constraint(matrix, rhs, mesh, bf, dh):
    """
    Convert the first two rows of the matrix and right hand side
    into a constraint that enforces a total solution in both the x and y
    direction of 0.
    """
    # Set the rhs to 0 for the first two rows
    first_x_dof = dh.dof_map[0, 0, 0]
    rhs[first_x_dof] = 0
    first_y_dof = dh.dof_map[1, 0, 0]
    rhs[first_y_dof] = 0

    quad_info = QuadGauss(bf.num_fncs).quad_info
    kernel = MassMatrixKernel(0, 0)
    one = ConstantEval([1.0, 1.0])
    for k in range(mesh.n_elements):
        for i in range(bf.num_fncs):
            dof_x = dh.dof_map[0, k, i]
            dof_y = dh.dof_map[1, k, i]
            integral = single_integral(mesh.mesh_eval,
                                       kernel,
                                       bf._basis_eval,
                                       one,
                                       quad_info,
                                       k, i, 0)
            matrix[first_x_dof, dof_x] += integral[0][0]
            matrix[first_y_dof, dof_y] += integral[1][1]

# TODO: Refactor this into a "pin-element" constraint that takes an
# element id as an input and a left-right flag
def pin_ends_constraint(matrix, rhs, left_end, right_end, dh):
    """
    Pin the values at the ends of a simple line mesh. This is useful
    for crack solutions with only traction boundary conditions, because
    the matrix is singular. Pinning the ends removes the rigid body motion
    modes of solution and results in a nonsingular matrix.
    """
    first_x_dof = dh.dof_map[0, 0, 0]
    last_x_dof = dh.dof_map[0, -1, -1]
    first_y_dof = dh.dof_map[1, 0, 0]
    last_y_dof = dh.dof_map[1, -1, -1]
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
