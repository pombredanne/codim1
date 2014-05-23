import numpy as np
from codim1.core import *
from codim1.fast_lib import rl_single_integral, TractionKernel,\
                            single_integral, ConstantBasis

def test_basis_quad_aligned_ordering():
    bf = gll_basis(10)
    quad_info = rl1_quad(10, 0, -5)
    np.testing.assert_almost_equal(np.array(bf.nodes), np.array(quad_info.x))

def test_rl_integral():
    N = 1
    mesh = simple_line_mesh(1, (0, 0), (1, 0))
    mapping = PolynomialMapping(mesh.elements[0])
    kernel = TractionKernel(1.0, 0.25)
    bf = basis_from_degree(0)
    one = ConstantBasis(np.ones(2))

    quad_info_old = gauss(1)
    quad_info_new = rl1_quad(1, 0.0, 5.0)

    pt = [0.0, -5.0]
    normal = [0.0, 1.0]
    kernel.set_interior_data(pt, normal)

    correct = single_integral(mapping.eval,
                              kernel, one, bf, quad_info_old, 0, 0)
    new_quadrature = single_integral(mapping.eval,
                              kernel, one, bf, quad_info_new, 0, 0)
    correct = np.array(correct)
    new_quadrature = np.array(new_quadrature)
    np.testing.assert_almost_equal(correct, new_quadrature)
    # new_version = rl_single_integral(mapping.eval, kernel, bf,
    #                                  quad_info_new, 0)
