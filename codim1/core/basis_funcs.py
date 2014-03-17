import numpy as np
import scipy.interpolate as spi
import copy
from codim1.fast.basis_funcs import evaluate_basis as _evaluate_basis

class BasisFunctions(object):
    """
        This class handles interactions with Lagrange polynomials defined on
        the unit reference interval [0, 1].
    """

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

    def evaluate_basis(self, i, x):
        return _evaluate_basis(self.fncs, i, x)

    def evaluate_basis_derivative(self, i, x):
        """
            Evaluates the derivative of the i-th lagrange polynomial at x
        """
        if self.num_fncs == 1:
            return 0.0
        retval = 0.0
        for c_idx, c  in enumerate(self.derivs[i]):
            retval += c * x ** (self.num_fncs - 2 - c_idx)
        return retval

    def times_coeffs(self, x, C):
        """
            Evaluates the sum of the lagrange polynomials times their
            respective coefficients -- C.
            C should be a numpy array
        """
        return C * np.array([self.evaluate_basis(i, x) for i in
                             range(len(self.nodes))])

class _DerivativeBasisFunctions(BasisFunctions):
    """
    Stores the derivatives of the basis functions. For internal use only.
    """
    def __init__(self, nodes, fncs):
        self.num_fncs = len(fncs)
        self.nodes = nodes
        self.fncs = fncs
