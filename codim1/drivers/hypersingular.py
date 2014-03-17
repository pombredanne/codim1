import numpy as np
import matplotlib.pyplot as plt
from codim1.core.dof_handler import ContinuousDOFHandler
from codim1.core.mesh import Mesh
from codim1.core.assembler import Assembler
from codim1.core.basis_funcs import BasisFunctions
from codim1.fast.elastic_kernel import ElastostaticKernel
from codim1.core.quad_strategy import QuadStrategy
from codim1.core.quadrature import QuadGauss
from codim1.core.mass_matrix import MassMatrix
from codim1.core.interior_point import InteriorPoint
import codim1.core.tools as tools

# This file solves a Neumann problem using the hypersingular boundary
# integral equation -- also known as the traction boundary integral
# equation, because all the terms are in traction units.

# Elastic parameters
shear_modulus = 1.0
poisson_ratio = 0.25

# Quadrature points for the various circumstances
quad_min = 4
quad_max = 12
quad_logr = 12
quad_oneoverr = 12

bf = BasisFunctions.from_degree(1)
mesh = Mesh.simple_line_mesh(50)
qs = QuadStrategy(mesh, quad_min, quad_max, quad_logr, quad_oneoverr)
tools.plot_mesh(mesh)
plt.show()
kernel = ElastostaticKernel(shear_modulus, poisson_ratio)
dh = ContinuousDOFHandler(mesh, 1)
assembler = Assembler(mesh, bf, dh, qs)
mass_matrix = MassMatrix(mesh, bf, dh, QuadGauss(2),
                         compute_on_init = True)

print('Assembling displacement->traction kernel matrix, Gpu')
Gpu = assembler.assemble_matrix(kernel.traction_adjoint, 'oneoverr')
print('Assembling traction->traction kernel matrix, Gpp')
Gpp = assembler.assemble_matrix(kernel.hypersingular_regularized, 'logr')
