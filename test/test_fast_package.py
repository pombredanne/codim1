import numpy as np
import codim1.fast_lib as fp
from codim1.core import *
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
def test_fast_dbl_integral():
    msh = simple_line_mesh(5)
    bf = basis_from_degree(2)
    k_d = fp.DisplacementKernel(1.0, 0.25)
    qs = QuadStrategy(msh, 10, 10, 10, 10)
    o_q, i_q = qs.get_quadrature(k_d.singularity_type,
                    msh.elements[0], msh.elements[4])
    def time_dbl_integral():
        for i in range(10000):
            a = fp.double_integral(msh.elements[0].mapping.eval,
                               msh.elements[4].mapping.eval,
                               k_d, bf, bf, o_q, i_q,
                               0, 0)
    print_extime(time_dbl_integral)
