import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections

def L2_error(f1, f2):
    L2_f2 = np.sqrt(np.sum((f2 ** 2)))
    L2_f1_minus_f2 = np.sqrt(np.sum(((f1 - f2) ** 2)))
    return L2_f1_minus_f2 / L2_f2

def default_preprocess(ref_pts, pts, e):
    return pts, dict()

def plot_mesh(msh, show = True, points_per_element = 5, fig_ax = None,
        preprocess = default_preprocess):
    """
    This function walks over the edges of a mesh and produces plotted lines from
    each of those edges.
    preprocess = a function can be specified to handle the reference and
    physical coordinates before actually plotting them. This allows
    more flexibility like changing the width or color of lines.
    """
    # Get the figure and axes on which to actually plot the mesh
    if fig_ax is None:
        fig, ax = plt.subplots()
    else:
        fig = fig_ax[0]
        ax = fig_ax[1]

    # We need points_per_element local points.
    x_hat = np.linspace(0.0, 1.0, points_per_element)

    all_lc_info = dict()
    all_pts = []
    all_lines = []
    for k in range(msh.n_elements):
        e = msh.elements[k]

        # We want to plot physical points
        points = map(e.mapping.get_physical_point, x_hat)

        # Process points into the desired form
        # (for example, convert from m to km)
        lc_info = preprocess(x_hat, points, e)
        for k,v in lc_info[1].iteritems():
            if k not in all_lc_info:
                all_lc_info[k] = v
            else:
                all_lc_info[k].extend(v)
        lines = zip(lc_info[0][:-1], lc_info[0][1:])
        all_pts.extend(lc_info[0])
        all_lines.extend(lines)

    # Create and add the line collection
    lc = matplotlib.collections.LineCollection(all_lines, **all_lc_info)
    ax.add_collection(lc)

    ax.autoscale()
    ax.margins(0.1)
    if show:
        fig.show()
    return fig, ax

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

def local_interpolate(fnc, mapping, basis):
    local_coeffs = np.empty((2, len(basis.nodes)))
    for i, n in enumerate(basis.nodes):
        node_pt = mapping.get_physical_point(n)
        normal = mapping.get_normal(n)
        f_val = fnc(node_pt, normal)
        local_coeffs[0, i] = f_val[0]
        local_coeffs[1, i] = f_val[1]
    return local_coeffs

def interpolate(fnc, mesh):
    """
    Interpolate the value of fnc onto the polynomial basis function space
    defined by basis_funcs and mesh. This simply sets the coefficient of each
    basis function to be the value of fnc at the node (where phi(x) = 1).
    """
    result = np.empty(mesh.total_dofs)
    for e_k in mesh:
        result[e_k.dofs] = local_interpolate(fnc, e_k.mapping, e_k.basis)
    return result
