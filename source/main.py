import numpy as np
from basis_funcs import BasisFunctions
from quadrature import QuadGauss
from mesh import Mesh
from elastic_kernel import ElastostaticKernel

# Number of elements
n_elements = 10

# Degree of the polynomial basis to use. For example, 1 is a linear basis
element_deg = 1

# Dimension of problem
dim = 2

q = QuadGauss(element_deg + 1)
bf = BasisFunctions(q.x)
mesh = Mesh.simple_line_mesh(n_elements)

# element_deg + 1 degrees of freedom per element
total_dofs = n_elements * (element_deg + 1)

# Making a dof_map in 1D is super simple! Its just a folded over list of all
# the dofs
dof_map = np.arange(total_dofs).reshape(n_elements, element_deg + 1)

# Specify which boundary conditions we have for each element
# Traction for every element with t_x = 0, t_y = 1.0
fnc = lambda x, y: (0.0, 1.0)
bc_list = [('traction', fnc) for i in range(n_elements)]

# See section 2.7 of starfield and crouch for the standard formulas to convert
# from plane strain to plane stress.
shear_modulus = 30e9
poisson_ratio = 0.25
kernel = ElastostaticKernel(shear_modulus, poisson_ratio)

quad_points_nonsingular = 5
quad_points_logr = 5
quad_points_oneoverr = 5

def interpolate(self, element_id, fnc):
    """
    Interpolate a function of (x, y) onto
    """
    pass
