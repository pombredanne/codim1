import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections

# def plot_kernel(kernel_fnc,
#     x = np.linspace(0.0, 1.0, 100)
#     y = np.linspace(0.0, 1.0, 100)
#     X, Y = np.meshgrid(x, y)
#     a = np.zeros_like(X)
#     b = np.zeros_like(X)
#     for i in range(X.shape[0]):
#         for j in range(X.shape[1]):
#             a[i, j] = f(x[i], y[j])[1, 1]
#             b[i, j] = ff(x[i], y[j])
#     import matplotlib.pyplot as plt
#     plt.figure()
#     plt.imshow(a)
#     plt.colorbar()
#     plt.figure()
#     plt.imshow(b)
#     plt.colorbar()
#     plt.show()

def L2_error(f1, f2):
    L2_f2 = np.sqrt(np.sum((f2 ** 2)))
    L2_f1_minus_f2 = np.sqrt(np.sum(((f1 - f2) ** 2)))
    return L2_f1_minus_f2 / L2_f2

def plot_mesh(msh, show = True, points_per_element = 5):
    points = []
    x_hat = np.linspace(0, 1, points_per_element)
    fig, ax = plt.subplots()
    for k in range(msh.n_elements):
        points = []
        e = msh.elements[k]
        for i in range(points_per_element):
            points.append(e.mapping.get_physical_point(x_hat[i]))
        lc = matplotlib.collections.LineCollection(
                            zip(points[:-1], points[1:]))
        ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    if show:
        fig.show()

def plot_matrix(M, title, show = True):
    """
    Creates and shows two plots. One of the matrix itself and
    one of the symmetric pattern M - M.T
    """
    # TODO: I think there's a matplotlib function that plots matrices and
    # probably does a much nicer job than what I have here.
    plt.figure()
    plt.imshow(M)
    plt.title(title)
    plt.colorbar()

    plt.figure()
    plt.imshow((M - M.T) / (M + M.T))
    plt.title(title)
    plt.colorbar()

    if show:
        plt.show()


def interpolate(fnc, mesh):
    """
    Interpolate the value of fnc onto the polynomial basis function space
    defined by basis_funcs and mesh. This simply sets the coefficient of each
    basis function to be the value of fnc at the node (where phi(x) = 1).
    """
    result = np.empty(mesh.total_dofs)
    for k in range(mesh.n_elements):
        e = mesh.elements[k]
        for i in range(e.basis.num_fncs):
            dof_x = e.dofs[0, i]
            dof_y = e.dofs[1, i]
            ref_pt = e.basis.nodes[i]
            node_pt = e.mapping.get_physical_point(ref_pt)
            normal = e.mapping.get_normal(ref_pt)
            f_val = fnc(node_pt, normal)
            result[dof_x] = f_val[0]
            result[dof_y] = f_val[1]
    return result

def evaluate_boundary_solution(points_per_element, soln, mesh):
    """
    Once a solution is computed, it's often nice to know the actual value, not
    just the coefficients of the polynomial basis! This function will
    produce points_per_elements points of the solution per element. Whee!

    This accepts a vector-valued solution and produces vector-valued point
    evaluations. A scalar version would be simple to write with this as a
    template.
    """
    x = []
    y = []
    for k in range(mesh.n_elements):
        e = mesh.elements[k]
        for pt in np.linspace(0.0, 1.0, points_per_element):
            x.append(e.mapping.get_physical_point(pt))
            u = evaluate_solution_on_element(e, pt, soln)
            y.append(u)
    x = np.array(x)
    y = np.array(y)
    return x, y

def evaluate_solution_on_element(element, reference_point, soln):
    phys_pt = element.mapping.get_physical_point(reference_point)
    soln_x = 0.0
    soln_y = 0.0
    # The value is the sum over all the basis functions.
    for i in range(element.basis.num_fncs):
        dof_x = element.dofs[0, i]
        dof_y = element.dofs[1, i]
        # TODO: remove element_idx from basis_funcs!
        soln_x += soln[dof_x] * element.basis.evaluate(i, reference_point, phys_pt)[0]
        soln_y += soln[dof_y] * element.basis.evaluate(i, reference_point, phys_pt)[1]
    return np.array([soln_x, soln_y])

# def interior_stresses(lower_left, upper_right, n_pts_per_dim,
#                       mesh, dh, qs, k_ta, k_h):
#     if type(n_pts_per_dim) == int:
#         n_pts_per_dim = (n_pts_per_dim, n_pts_per_dim)
