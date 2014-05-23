"""
Thin wrappers over the quadracheer package that convert from [-1, 1] to [0, 1]
"""
import numpy as np
import quadracheer as qc
from codim1.fast_lib import QuadratureInfo

def gauss(N):
    x, w = qc.map_nonsing(qc.gaussxw, N, 0.0, 1.0)
    return QuadratureInfo(0.0, x, w)

def piessens(N, x0, nonsingular_N):
    x, w = qc.map_singular(qc.piessens, N, x0, 0.0, 1.0,
                           nonsingular_N = nonsingular_N)
    return QuadratureInfo(x0, x, w)

def telles_singular(N, x0):
    x, w = qc.map_singular(qc.telles_singular, N, x0, 0.0, 1.0)
    return QuadratureInfo(x0, x, w)

def telles_quasi_singular(N, x0, D):
    x, w = qc.map_singular(qc.telles_quasi_singular, N, x0, 0.0, 1.0, D = D)
    return QuadratureInfo(x0, x, w)

def rl1_quad(N, ay, by):
    x, w = qc.map_rl_quad(qc.rl_quad, N, ay, by, 0.0, 1.0)
    # Fix this
    new_w = w * np.sqrt((x - ay) ** 2 + by ** 2)
    return QuadratureInfo(ay, x, new_w)
