"""
Thin wrappers over the quadracheer package that convert from [-1, 1] to [0, 1]
"""
import quadracheer as qc
from codim1.fast_lib import QuadratureInfo

def gauss(N):
    x, w = qc.map_nonsing(qc.gaussxw, N, 0, 1)
    return QuadratureInfo(0.0, x, w)

def piessens(N, x0, nonsingular_N):
    x, w = qc.map_singular(qc.piessens, N, x0, 0, 1,
                           nonsingular_N = nonsingular_N)
    return QuadratureInfo(x0, x, w)

def telles_singular(N, x0):
    x, w = qc.map_singular(qc.telles_singular, N, x0, 0, 1)
    return QuadratureInfo(x0, x, w)

def telles_quasi_singular(N, x0, D):
    x, w = qc.map_singular(qc.telles_quasi_singular, N, x0, 0, 1, D = D)
    return QuadratureInfo(x0, x, w)
