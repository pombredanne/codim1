import numpy as np
import codim1.core.basis_funcs as basis_funcs
from codim1.fast_lib import HypersingularKernel, AdjointTractionKernel
import codim1.core.mesh as mesh
import codim1.core.dof_handler as dof_handler
from codim1.core.quad_strategy import QuadStrategy
from codim1.core.interior_point import InteriorPoint

# Some ugly but real world tests. Couldn't think of anything simpler to try it
# on....
def test_interior_point_hypersingular():
    msh = mesh.Mesh.circular_mesh(10, 1.0)
    bf = basis_funcs.BasisFunctions.from_degree(0)
    dh = dof_handler.DiscontinuousDOFHandler(msh, bf)
    qs = QuadStrategy(msh, 6, 12, 2, 2)
    coeffs = np.array([ -1.51048858e-02,  -6.11343409e-03,  -1.97679606e-11,
         6.11343405e-03,   1.51048857e-02,   1.51048857e-02,
         6.11343405e-03,  -1.97684512e-11,  -6.11343409e-03,
        -1.51048858e-02,   3.29088618e-03,   1.41159450e-02,
         4.68710442e-02,   1.41159450e-02,   3.29088618e-03,
        -3.29088619e-03,  -1.41159451e-02,  -4.68710442e-02,
        -1.41159451e-02,  -3.29088619e-03])
    soln = basis_funcs.Solution(bf, dh, coeffs)
    kernel = HypersingularKernel(1.0, 0.25)
    ip = InteriorPoint(msh, dh, qs)
    result = ip.compute(np.array([0.5, 0.0]), np.array([1.0, 0.0]),
                        kernel, soln)
    np.testing.assert_almost_equal(result[0], -0.01064342343)
    np.testing.assert_almost_equal(result[1], 0.0)

def test_interior_point_traction_adjoint():
    msh = mesh.Mesh.circular_mesh(50, 1.0)
    bf = basis_funcs.BasisFunctions.from_degree(0)
    dh = dof_handler.DiscontinuousDOFHandler(msh, bf)
    qs = QuadStrategy(msh, 6, 12, 2, 2)
    def section_traction(x, d):
        if np.abs(x[0]) < np.cos(24 * (np.pi / 50)):
            x_length = np.sqrt(x[0] ** 2 + x[1] ** 2)
            return -x[d] / x_length
        return 0.0
    traction_function = basis_funcs.BasisFunctions.\
            from_function(section_traction)
    kernel = AdjointTractionKernel(1.0, 0.25)
    ip = InteriorPoint(msh, dh, qs)
    result = ip.compute(np.array([0.5, 0.0]), np.array([1.0, 0.0]), kernel,
                traction_function)
    np.testing.assert_almost_equal(result[0], -0.002094,4)
    np.testing.assert_almost_equal(result[1], 0.0)

if __name__ == "__main__":
    test_interior_point_traction_adjoint()
