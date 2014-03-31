import numpy as np
import codim1.fast_lib as fp
from codim1.core.quad_strategy import QuadStrategy
from codim1.core.basis_funcs import BasisFunctions
from codim1.core.mesh import Mesh
from time import time
import pytest

def print_extime(f):
    start = time()
    f()
    print "%s: %fs" % (f.__name__, time() - start)

def nothing_inner():
    a = 0
def nothing_outer():
    for i in range(1000000):
        nothing_inner()

@pytest.mark.slow
def test_fast_basis_eval():
    myarray = np.random.rand(2, 2)
    bf = BasisFunctions.from_degree(2)
    be = fp.BasisEval(bf.fncs)

    def one():
        for i in range(1000000):
            bf.evaluate(1, 1, 0.5, 0.5)

    def two():
        for i in range(1000000):
            be.evaluate(1, 0.5)

    print_extime(one)
    print_extime(two)
    print_extime(nothing_outer)

    print_extime(lambda: fp.basis_speed_test(bf.fncs))

@pytest.mark.slow
def test_fast_dbl_integral():
    msh = Mesh.simple_line_mesh(5)
    bf = BasisFunctions.from_degree(2)
    k_d = fp.DisplacementKernel(1.0, 0.25)
    qs = QuadStrategy(msh, 10, 10, 10, 10)
    o_q, i_q = qs.get_quadrature(k_d.singularity_type, 0, 4)
    o_qi = o_q.quad_info
    i_qi = [iq.quad_info for iq in i_q]
    def time_dbl_integral():
        for i in range(10000):
            a = fp.double_integral(msh.mesh_eval, True, k_d,
                               bf._basis_eval, bf._basis_eval,
                               o_qi, i_qi, 0, 0, 4, 0)
    print_extime(time_dbl_integral)
