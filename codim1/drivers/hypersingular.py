import numpy as np
import matplotlib.pyplot as plt

from codim1.assembly import *
from codim1.core import *
import codim1.core.tools as tools
from codim1.fast_lib import AdjointTractionKernel,\
                                       RegularizedHypersingularKernel,\
                                       DisplacementKernel,\
                                       TractionKernel

# This file solves a Neumann problem using the hypersingular boundary
# integral equation -- also known as the traction boundary integral
# equation, the boundary conditions are tractions

# Elastic parameters
shear_modulus = 1.0
poisson_ratio = 0.25

# Quadrature points for the various circumstances
quad_min = 4
quad_max = 12
quad_logr = 12
quad_oneoverr = 12
interior_quad_pts = 8

n_elements = 80

k_d = DisplacementKernel(shear_modulus, poisson_ratio)
k_t = TractionKernel(shear_modulus, poisson_ratio)
k_tp = AdjointTractionKernel(shear_modulus, poisson_ratio)
k_h = RegularizedHypersingularKernel(shear_modulus, poisson_ratio)


# The standard structures for a problem.
mesh = Mesh.simple_line_mesh(n_elements)
bf = BasisFunctions.from_degree(1)
qs = QuadStrategy(mesh, quad_min, quad_max, quad_logr, quad_oneoverr)
dh = DOFHandler(mesh, bf)
assembler = MatrixAssembler(mesh, bf, dh, qs)

# Build the rhs
def fnc(x, d):
    if d == 1:
        return 0.0
    return 1.0
traction_func = BasisFunctions.from_function(fnc)
mass_matrix = MassMatrix(mesh, bf, traction_func, dh,
                         QuadGauss(3), compute_on_init = True)
rhs = -mass_matrix.for_rhs()

print('Assembling kernel matrix, Gpp')
# Use the basis function arclength derivatives for the regularized
# Gpp (hypersingular) kernel
# This is derived using integration by parts and moving part of the 1/r^3
# singularity onto the basis functions.
assembler = MatrixAssembler(mesh, bf, dh, qs)
derivs_assembler = MatrixAssembler(mesh, bf.get_gradient_basis(mesh), dh, qs)
Guu = assembler.assemble_matrix(k_d)
Gup = assembler.assemble_matrix(k_t)
Gpu = Gup.T
Gpp = derivs_assembler.assemble_matrix(k_h)

matrix = np.zeros_like(Gpp)
bc_type = np.zeros(n_elements)
bc_type[0] = 1
bc_type[-1] = 1
for k in range(n_elements):
    for i in range(bf.num_fncs):
        for l in range(n_elements):
            for j in range(bf.num_fncs):
                dof_k = dh.dof_map[:, k, i]
                dof_l = dh.dof_map[:, l, j]
                if matrix[dof_k[0], dof_l[0]] != 0.0:
                    continue
                if bc_type[k] == 0 and bc_type[l] == 0:
                    local_matrix = Gpp
                if bc_type[k] == 0 and bc_type[l] == 1:
                    local_matrix = Gpu
                if bc_type[k] == 1 and bc_type[l] == 0:
                    local_matrix = Gup
                    rhs[dof_k] = 0.0
                if bc_type[k] == 1 and bc_type[l] == 1:
                    local_matrix = Guu
                    rhs[dof_k] = 0.0
                for d1 in range(2):
                    for d2 in range(2):
                        matrix[dof_k[d1], dof_l[d2]] = \
                            local_matrix[dof_k[d1], dof_l[d2]]

soln_coeffs = np.linalg.solve(matrix, rhs)

# Create a solution object that pairs the coefficients with the basis
soln = Solution(bf, dh, soln_coeffs)
x, s = tools.evaluate_boundary_solution(5, soln, mesh)
# s[:5, :] = 0.0

plt.figure(1)
plt.plot(x[:, 0], s[:, 0])
plt.xlabel(r'X')
plt.ylabel(r'$u_x$', fontsize = 18)
plt.figure(2)
plt.plot(x[:, 0], s[:, 1])
plt.xlabel(r'X')
plt.ylabel(r'$u_y$', fontsize = 18)
plt.show()

import sys
sys.exit()

# Compute some interior values.
x_pts = 40
y_pts = 40
x = np.linspace(-5, 5, x_pts)
# Doesn't sample 0.0!
y = np.linspace(-5, 5, y_pts)
int_ux = np.zeros((x_pts, y_pts))
int_uy = np.zeros((x_pts, y_pts))
ip = InteriorPoint(mesh, dh, qs)
for i in range(x_pts):
    for j in range(y_pts):
        traction_effect = ip.compute((x[i], y[j]), np.array((0.0, 0.0)),
                   k_d, traction_func)
        displacement_effect = -ip.compute((x[i], y[j]), np.array([0.0, 0.0]),
                   k_t, soln)
        int_ux[j, i] = traction_effect[0] + displacement_effect[0]
        int_uy[j, i] = traction_effect[1] + displacement_effect[1]

plt.figure(3)
plt.title(r'$u_x$')
plt.imshow(int_ux)
plt.colorbar()
plt.figure(4)
plt.title(r'$u_y$')
plt.imshow(int_uy)
plt.colorbar()
plt.show()
import ipdb;ipdb.set_trace()
