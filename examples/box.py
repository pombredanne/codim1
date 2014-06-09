"""
Apply a displacement to the top of the box and zero traction to the
sides. Plot the displacement and stress fields.
"""
from codim1.core import *
from codim1.fast_lib import *
from codim1.assembly import *

import numpy as np
import matplotlib.pyplot as plt
from functools import partial

shear_modulus = 1.0
poisson_ratio = 0.25
n_elements_per_side = 10
degree = 2

ek = ElasticKernelSet(shear_modulus, poisson_ratio)

bc_free_surf = BC("traction", ZeroBasis())
bc_down = BC("displacement", ConstantBasis([0.0, -1.0]))
bc_fixed = BC("displacement", ZeroBasis())
mesh = rect_mesh(n_elements_per_side, (-1, 1), (1, -1),
                {
                    'top': bc_down,
                    'left': bc_free_surf,
                    'right': bc_free_surf,
                    'bottom': bc_fixed
                })
# tools.plot_mesh(mesh)

qs = QuadStrategy(mesh, 10, 10, 12, 12)
bf = basis_from_degree(degree)
apply_to_elements(mesh, "basis", bf, non_gen = True)
apply_to_elements(mesh, "continuous", True, non_gen = True)
apply_to_elements(mesh, "qs", qs, non_gen = True)

total_dofs = init_dofs(mesh)

# Assemble the system
# The critical new function sgbem_assembler!
matrix, rhs = sgbem_assemble(mesh, ek)

# Solve the system
soln_coeffs = np.linalg.solve(matrix, rhs)

# Need a new function here.
x, u, t = evaluate_sgbem_solution(5, mesh, soln_coeffs)

plt.figure()
plt.plot(x[1, :], u[0, :])
plt.figure()
plt.plot(x[1, :], u[1, :])
plt.show()

# Make some plots
