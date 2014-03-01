import numpy as np
from basis_funcs import BasisFunctions
from quadrature import QuadGauss
from mesh import Mesh

# Number of elements
n_elements = 10

# Degree of the polynomial basis to use. For example, 1 is a linear basis
element_deg = 1

q = QuadGauss(2)
bf = BasisFunctions(x)
mesh = Mesh.simple_line_mesh(n_elements)

# element_deg + 1 degrees of freedom per element
total_dofs = n_elements * (element_deg + 1)
# Making a dof_map in 1D is super simple! Its just a folded over list of all
# the dofs
dof_map = np.arange(total_dofs).reshape(n_elements, element_deg + 1)

# See section 2.7 of starfield and crouch for the standard formulas to convert
# from plane strain to plane stress.
youngs = 30e9
poisson = 0.25

# Input tractions
sigma_n = np.ones(n_elements)
sigma_s = np.zeros(n_elements)
