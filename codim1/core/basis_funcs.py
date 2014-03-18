import numpy as np
import scipy.interpolate as spi
import copy
from codim1.fast.basis_funcs import evaluate_basis as _evaluate_basis

class Function(object):
    def __init__(self, f):
        self.f = f

    def evaluate(self, i, x_hat, x):
        return self.f(x)

    def chain_rule(self, element_idx):
        return 1.0

# class Solution(object):
#     def __init__(self, basis, coefficients):
#         self.basis = basis
#         self.coefficients = coefficients
#
#     def evaluate(self, i, x_hat, x):

class BasisFunctions(object):
    """
        This class handles interactions with Lagrange polynomials defined on
        the unit reference interval [0, 1].
        The basis functions are defined such that
        \hat{\phi}_i(\hat{x}) = \phi_i(x)
        In other words, there is no transformation factor between reference
        and physical space. So, there is no chain rule contribution.
    """
    @classmethod
    def from_function(cls, f):
        return Function(f)

    @classmethod
    def from_degree(cls, element_deg, mesh):
        """
            Create an equispaced nodal basis.
        """
        if element_deg == 0:
            nodes = np.array([0.5])
        else:
            nodes = np.linspace(0.0, 1.0, element_deg + 1)
        return cls(nodes, mesh)

    def __init__(self, nodes, mesh):
        """
            Builds the Lagrange interpolating polynomials with nodes at the
            points specified.
        """
        self.mesh = mesh
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
        self.derivs = _DerivativeBasisFunctions(self.nodes, derivs, mesh)

    def evaluate(self, i, x_hat, x):
        return _evaluate_basis(self.fncs, i, x_hat)

    def chain_rule(self, element_idx):
        return 1.0

class _DerivativeBasisFunctions(BasisFunctions):
    """
    Stores the derivatives of the basis functions. For internal use only.
    Because the derivative is now defined on the reference triangle, the
    transformation from physical to reference space requires an application
    of the chain rule. This gives an extra term d\hat{x}/dx. Thus, these
    basis functions cannot be defined except in reference to a specific
    mesh.
    """
    def __init__(self, nodes, fncs, mesh):
        self.mesh = mesh
        self.num_fncs = len(fncs)
        self.nodes = nodes
        self.fncs = fncs

    def chain_rule(self, element_idx):
        # In 1D, all we need is the inverse of the element jacobian
        # determinant -- which is just the single dx/d\hat{x}
        return 1.0 / self.mesh.get_element_jacobian(element_idx)
