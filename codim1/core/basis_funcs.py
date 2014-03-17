import numpy as np
import scipy.interpolate as spi
import copy
from codim1.fast.basis_funcs import evaluate_basis as _evaluate_basis

class Function(object):
    def __init__(self, f):
        self.f = f

    def evaluate_basis(self, i, x_hat, x):
        return self.f(x)

class BasisFunctions(object):
    """
        This class handles interactions with Lagrange polynomials defined on
        the unit reference interval [0, 1].
    """
    @classmethod
    def from_function(cls, f):
        return Function(f)

    @classmethod
    def from_degree(cls, element_deg):
        """
            Create an equispaced nodal basis.
        """
        if element_deg == 0:
            nodes = np.array([0.5])
        else:
            nodes = np.linspace(0.0, 1.0, element_deg + 1)
        return cls(nodes)

    def __init__(self, nodes):
        """
            Builds the Lagrange interpolating polynomials with nodes at the
            points specified.
        """
        self.num_fncs = len(nodes)
        self.fncs = np.empty((self.num_fncs, self.num_fncs))
        derivs = np.empty((self.num_fncs, self.num_fncs))
        self.nodes = copy.copy(nodes)
        for (i, n) in enumerate(nodes):
            w = np.zeros_like(nodes)
            w[i] = 1.0
            # scipy.interpolate.lagrange has trouble above 20 nodes, but that
            # shouldn't be an issue for this code
            poly = spi.lagrange(nodes, w)
            self.fncs[i, :] = poly.c
            derivs[i, 0] = 0.0
            derivs[i, 1:] = poly.deriv().c
        self.derivs = _DerivativeBasisFunctions(self.nodes, derivs)

    def evaluate_basis(self, i, x_hat, x):
        return _evaluate_basis(self.fncs, i, x_hat)

class _DerivativeBasisFunctions(BasisFunctions):
    """
    Stores the derivatives of the basis functions. For internal use only.
    """
    def __init__(self, nodes, fncs):
        self.num_fncs = len(fncs)
        self.nodes = nodes
        self.fncs = fncs
