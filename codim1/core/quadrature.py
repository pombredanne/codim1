"""
Thin wrappers over the quadracheer package that convert from [-1, 1] to [0, 1]
"""
import numpy as np
import quadracheer as qc
from codim1.fast_lib import QuadratureInfo

def gauss(N):
    x, w = qc.gaussxw(N)
    x, w = qc.map_pts_wts(x, w, 0.0, 1.0)
    return QuadratureInfo(0.0, x, w)

def lobatto(N):
    x, w = qc.lobatto_quad(N)
    x, w = qc.map_pts_wts(x, w, 0.0, 1.0)
    return QuadratureInfo(0.0, x, w)

def piessens(N, x0, nonsingular_N):
    mapped_x0 = qc.map_singular_pt(x0, 0.0, 1.0)
    x, w = qc.piessens(N, mapped_x0, nonsingular_N = nonsingular_N)
    x, w = qc.map_pts_wts(x, w, 0.0, 1.0)
    return QuadratureInfo(x0, x, w)

def telles_singular(N, x0):
    mapped_x0 = qc.map_singular_pt(x0, 0.0, 1.0)
    x, w = qc.telles_singular(N, mapped_x0)
    x, w = qc.map_pts_wts(x, w, 0.0, 1.0)
    return QuadratureInfo(x0, x, w)

def telles_quasi_singular(N, x0, D):
    # I think D needs to be mapped into local coords too
    mapped_x0 = qc.map_singular_pt(x0, 0.0, 1.0)
    x, w = qc.telles_quasi_singular(N, mapped_x0, D)
    x, w = qc.map_pts_wts(x, w, 0.0, 1.0)
    return QuadratureInfo(x0, x, w)

def rl_quad(N, ay, by):
    mapped_ay = qc.map_singular_pt(ay, 0.0, 1.0)
    mapped_by = qc.map_distance_to_interval(by, 0.0, 1.0)
    moments = qc.modified_moments(qc.rl1, N - 1, mapped_ay, mapped_by)
    x, w = qc.recursive_quad(moments)
    x, w = qc.map_pts_wts(x, w, 0.0, 1.0)
    w = qc.map_weights_by_inv_power(w, 1.0, 0.0, 1.0)

    # Fix this
    new_w = w * ((x - ay) ** 2 + by ** 2) ** (1.0 / 2.0)
    return QuadratureInfo(ay, x, new_w)
