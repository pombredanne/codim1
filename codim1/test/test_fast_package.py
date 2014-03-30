import numpy as np
import codim1.fast_lib as fp
from codim1.core.basis_funcs import BasisFunctions
from time import time

def print_extime(f):
    start = time()
    f()
    print "%s: %fs" % (f.__name__, time() - start)

def nothing_inner():
    a = 0
def nothing_outer():
    for i in range(1000000):
        nothing_inner()

def test_fast():
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

