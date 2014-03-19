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

def plot_mesh(msh, show = True):
    points1 = msh.vertices[msh.element_to_vertex[:, 0]]
    points2 = msh.vertices[msh.element_to_vertex[:, 1]]
    lc = matplotlib.collections.LineCollection(zip(points1, points2))
    fig, ax = plt.subplots()
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


def interpolate(fnc, dof_handler, basis_funcs, mesh):
    """
    Interpolate the value of fnc onto the polynomial basis function space
    defined by basis_funcs and mesh. This simply sets the coefficient of each
    basis function to be the value of fnc at the node (where phi(x) = 1).
    """
    result = np.empty(dof_handler.total_dofs)
    for k in range(mesh.n_elements):
        for i in range(basis_funcs.num_fncs):
            dof_x = dof_handler.dof_map[0, k, i]
            dof_y = dof_handler.dof_map[1, k, i]
            ref_pt = basis_funcs.nodes[i]
            node_pt = mesh.get_physical_points(k, ref_pt)
            normal = mesh.normals[k, :]
            f_val = fnc(node_pt, normal)
            result[dof_x] = f_val[0]
            result[dof_y] = f_val[1]
    return result

def evaluate_boundary_solution(points_per_element, soln, mesh):
    """
    Once a solution is computed, it's often nice to know the actual value, not
    just the coefficients of the polynomial basis! This function will produce
    1 point for every "point_separation" of distance along the boundary.

    This accepts a vector-valued solution and produces vector-valued point
    evaluations. A scalar version would be simple to write with this as a
    template.
    """
    x = []
    y = []
    for k in range(mesh.n_elements):
        for pt in np.linspace(0.0, 1.0, points_per_element):
            x.append(mesh.get_physical_points(k, pt))
            u = evaluate_solution_on_element(k, pt, soln, mesh)
            y.append(u)
    x = np.array(x)
    y = np.array(y)
    return x, y

def evaluate_solution_on_element(element_idx, reference_point, soln, mesh):
    phys_pt = mesh.get_physical_points(element_idx, reference_point)
    soln_x = 0.0
    soln_y = 0.0
    for i in range(soln.basis.num_fncs):
        soln_x += soln.evaluate(element_idx, i, reference_point, phys_pt)[0]
        soln_y += soln.evaluate(element_idx, i, reference_point, phys_pt)[1]
    return np.array([soln_x, soln_y])

