"""
Apply a displacement to the top of the box and zero traction to the
sides. Plot the displacement and stress fields.
"""
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from codim1 import *

shear_modulus = 1.0
poisson_ratio = 0.25
n_elements_per_side = 10
degree = 2

ek = ElasticKernelSet(shear_modulus, poisson_ratio)

bc_free_surf = partial(ZeroBC, "trac")
bc_down = partial(ConstantBC, "disp", 1.0)
bc_fixed = partial(ZeroBC, "disp")
mesh = rect_mesh(n_elements_per_side, (-1, 1), (1, -1),
                {
                    'top': bc_down,
                    'left': bc_free_surf,
                    'right': bc_free_surf,
                    'bottom': bc_fixed
                })
tools.plot_mesh(mesh)

apply_to_elements(mesh, "basis",
                  BasisFunctions.from_degree(degree),
                  non_gen = True)
apply_to_elements(mesh, "continuous", True, non_gen = True)

total_dofs = init_dofs(mesh)

# Assemble the system
qs = QuadStrategy(mesh, 10, 10, 11, 11)
# The critical new function sgbem_assembler!
matrix, rhs = sgbem_assembler(mesh, qs, ek)

# Solve the system
soln_coeffs = np.linalg.solve(matrix, rhs)

# Need a new function here.
# x, t, u = evaluate_sgbem_solution(pts_per_element, mesh, soln_coeffs)

# Make some plots
