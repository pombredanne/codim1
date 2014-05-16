import numpy as np
import scipy.interpolate as spi
import copy
from codim1.fast_lib import PolyBasisEval, FuncEval,\
                            GradientBasisEval, SolutionEval

"""
This stuff should be reworked so that basis functions exist elementwise.

Use another function like apply_mapping that applies a basis function to each
member of an iterable.
"""

class Function(object):
    """
    A thin wrapper around a normal python function in order to trick all
    the integration procedures into thinking it is a basis function.
    """
    def __init__(self, f):
        self.f = f
        self.num_fncs = 1
        self._basis_eval = FuncEval(f)

    def evaluate(self, element_idx, i, x_hat, x):
        return self.f(x)

class Solution(object):
    """
    Represents a solution by some coefficients multiplied by a set of
    basis functions defined on the reference element
    """
    def __init__(self, basis, dof_handler, coeffs):
        self.dof_handler = dof_handler
        self.basis = basis
        self.coeffs = coeffs
        self.num_fncs = basis.num_fncs
        self._basis_eval = SolutionEval(self.basis._basis_eval, self.coeffs,
                                        self.dof_handler.dof_map)

    def evaluate(self, element_idx, i, x_hat, x):
        val_x = self._basis_eval.evaluate(element_idx, i, x_hat, x, 0)
        val_y = self._basis_eval.evaluate(element_idx, i, x_hat, x, 1)
        return np.array([val_x, val_y])

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
        self.nodes = copy.copy(nodes)
        self.fncs = np.empty((self.num_fncs, self.num_fncs))
        self.derivs = np.empty((self.num_fncs, self.num_fncs))
        for (i, n) in enumerate(nodes):
            w = np.zeros_like(nodes)
            w[i] = 1.0
            # scipy.interpolate.lagrange has trouble above 20 nodes, but that
            # shouldn't be an issue for this code
            poly = spi.lagrange(nodes, w)
            self.fncs[i, :] = poly.c
            self.derivs[i, 0] = 0.0
            self.derivs[i, 1:] = poly.deriv().c
        self._basis_eval = PolyBasisEval(self.fncs)
        self._deriv_eval = PolyBasisEval(self.derivs)

    def get_gradient_basis(self, mesh):
        return _GradientBasisFunctions(self.nodes, self.derivs, mesh)

    def evaluate(self, element_idx, i, x_hat, x):
        """
        Evaluate a basis function at a point in reference space.

        Calls into the fast c++ extension.
        """
        val = self._basis_eval.evaluate(element_idx, i, x_hat, [0, 0], 0)
        return np.array([val, val])

    def chain_rule(self, jacobian):
        return self._basis_eval.chain_rule(jacobian)

    def evaluate_derivative(self, element_idx, i, x_hat, x):
        """
        Evaluate the derivative in reference space. Note that this does not
        include the chain rule term that would arise if the derivative were
        taken in physical space. Use _GradientBasisFunctions for that
        purpose.
        """
        val = self._deriv_eval.evaluate(element_idx, i, x_hat, [0, 0], 0)
        return np.array([val, val])

class _GradientBasisFunctions(BasisFunctions):
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
        self._basis_eval = GradientBasisEval(self.fncs)
