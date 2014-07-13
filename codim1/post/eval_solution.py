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

def evaluate_solution_on_element(element,
                                 reference_point,
                                 soln_coeffs,
                                 side = 'positive'):


    mult = 1.0
    if side == 'negative':
        mult = -1.0

    if element.bc.type == "displacement":
        u_info = (element.bc.basis, np.ones((2, element.bc.basis.n_fncs)), 1.0)
        t_info = (element.basis, soln_coeffs[element.dofs], mult)
    elif element.bc.type == "traction":
        u_info = (element.basis, soln_coeffs[element.dofs], 1.0)
        t_info = (element.bc.basis, np.ones((2, element.bc.basis.n_fncs)), mult)
    elif element.bc.type == "crack_traction":
        u_info = (element.basis, soln_coeffs[element.dofs], 0.5 * mult)
        t_info = (element.bc.basis, np.ones((2, element.bc.basis.n_fncs)), mult)
    elif element.bc.type == "crack_displacement":
        u_info = (element.bc.basis, np.ones((2, element.bc.basis.n_fncs)), 0.5 * mult)
        t_info = (element.basis, soln_coeffs[element.dofs], mult)

    u = eval_basis(element, u_info[0], reference_point, u_info[1]) * u_info[2]
    d_u = eval_basis(element, u_info[0].get_gradient_basis(),
                     reference_point, u_info[1]) * u_info[2]
    t = eval_basis(element, t_info[0], reference_point, t_info[1]) * t_info[2]
    return u, t

def eval_basis(element, basis, reference_point, coeffs):
    y = np.zeros(2)
    for i in range(basis.n_fncs):
        basis_eval = basis.evaluate(i, reference_point)
        y[0] += coeffs[0, i] * basis_eval[0]
        y[1] += coeffs[1, i] * basis_eval[1]
    return y
