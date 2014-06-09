import numpy as np

def evaluate_boundary_solution(mesh, soln, points_per_element):
    x, u, t = ([], [], [])
    for e_k in mesh:
        for pt in np.linspace(0.0, 1.0, points_per_element):
            x.append(e_k.mapping.get_physical_point(pt))
            local_u, local_t = evaluate_solution_on_element(e_k, pt, soln)
            u.append(local_u)
            t.append(local_t)
    return np.array(x).T, np.array(u).T, np.array(t).T

def evaluate_solution_on_element(element, reference_point, soln_coeffs):
    soln = np.zeros(2)
    # The value is the sum over all the basis functions.
    for i in range(element.basis.n_fncs):
        dof_x = element.dofs[0, i]
        dof_y = element.dofs[1, i]
        basis_eval = element.basis.evaluate(i, reference_point)
        soln[0] += soln_coeffs[dof_x] * basis_eval[0]
        soln[1] += soln_coeffs[dof_y] * basis_eval[1]

    bc = np.zeros(2)
    for i in range(element.bc.basis.n_fncs):
        bc_eval = element.bc.basis.evaluate(i, reference_point)
        bc[0] += bc_eval[0]
        bc[1] += bc_eval[1]

    if element.bc.type == "displacement":
        u = bc
        t = soln
    elif element.bc.type == "traction":
        u = soln
        t = bc
    elif element.bc.type == "crack_traction":
        u = soln / 2
        t = bc
    elif element.bc.type == "displacement_discontinuity":
        u = bc / 2
        t = soln
    return u, t
