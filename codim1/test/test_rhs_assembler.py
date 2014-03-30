import numpy as np
from codim1.core.rhs_assembler import RHSAssembler
import codim1.core.basis_funcs as basis_funcs
import codim1.fast.elastic_kernel as elastic_kernel
import codim1.core.mesh as mesh
import codim1.core.dof_handler as dof_handler
import codim1.core.quad_strategy as quad_strategy

# A presumed-to-be correct matrix formed from the G_up kernel
correct_matrix = \
    np.array([[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
     -3.82120741e-04,   4.22000426e-02,   1.08527120e-02],
   [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
     -4.21978289e-02,   3.60822483e-16,   4.21978289e-02],
   [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
     -1.08527120e-02,  -4.22000426e-02,   3.82120741e-04],
   [  3.82120741e-04,  -4.22000426e-02,  -1.08527120e-02,
      0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
   [  4.21978289e-02,  -3.60822483e-16,  -4.21978289e-02,
      0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
   [  1.08527120e-02,   4.22000426e-02,  -3.82120741e-04,
      0.00000000e+00,   0.00000000e+00,   0.00000000e+00]])

def rhs_assembler():
    bf = basis_funcs.BasisFunctions.from_degree(1)
    msh = mesh.Mesh.simple_line_mesh(2, -1.0, 1.0)
    dh = dof_handler.ContinuousDOFHandler(msh, bf)
    qs = quad_strategy.QuadStrategy(msh, 10, 10, 10, 10)
    assembler = RHSAssembler(msh, bf, dh, qs)
    return assembler

def test_rhs_row_not_shared():
    a = rhs_assembler()
    kernel = elastic_kernel.TractionKernel(1.0, 0.25)
    row_correct_x = np.sum(correct_matrix[0, :])
    row_correct_y = np.sum(correct_matrix[3, :])

    f = lambda x: np.array((1.0, 1.0))
    # Make the function look like a basis function. It is one! The only one!
    fnc = basis_funcs.BasisFunctions.from_function(f)
    row_x, row_y = a.assemble_row(fnc, kernel, 0, 0)

    np.testing.assert_almost_equal(row_correct_x, row_x)
    np.testing.assert_almost_equal(row_correct_y, row_y)

def test_rhs_row_shared():
    a = rhs_assembler()
    kernel = elastic_kernel.TractionKernel(1.0, 0.25)
    row_correct_x = np.sum(correct_matrix[1, :])
    row_correct_y = np.sum(correct_matrix[4, :])

    f = lambda x: np.array((1.0, 1.0))
    # Make the function look like a basis function. It is one! The only one!
    fnc = basis_funcs.BasisFunctions.from_function(f)
    row_x, row_y = a.assemble_row(fnc, kernel, 0, 1)
    row_x2, row_y2 = a.assemble_row(fnc, kernel, 1, 0)
    row_x += row_x2
    row_y += row_y2

    np.testing.assert_almost_equal(row_correct_x, row_x)
    np.testing.assert_almost_equal(row_correct_y, row_y)


def test_rhs():
    a = rhs_assembler()
    kernel = elastic_kernel.TractionKernel(1.0, 0.25)
    # If we sum across rows, we should get the RHS value for
    # a function that is 1 everywhere
    rhs_correct = np.sum(correct_matrix, axis = 1)

    f = lambda x: np.array((1.0, 1.0))
    # Make the function look like a basis function. It is one! The only one!
    fnc = basis_funcs.BasisFunctions.from_function(f)
    rhs = a.assemble_rhs(fnc, kernel)
    np.testing.assert_almost_equal(rhs_correct, rhs)


