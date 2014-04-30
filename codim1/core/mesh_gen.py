import numpy as np
from mesh import Mesh
from basis_funcs import BasisFunctions

def simple_line_mesh(n_elements,
        left_edge = (-1.0, 0.0), right_edge = (1.0, 0.0)):
    """
    Create a mesh consisting of a line of elements starting at -1 and
    extending to +1 in x coordinate, y = 0.
    """
    if type(left_edge) is float:
        left_edge = (left_edge, 0.0)
    if type(right_edge) is float:
        right_edge = (right_edge, 0.0)
    n_vertices = n_elements + 1
    vertex_params = np.linspace(left_edge[0], right_edge[0], n_vertices)
    #slope
    m = (right_edge[1] - left_edge[1]) / (right_edge[0] - left_edge[0])
    c = right_edge[0]
    d = right_edge[1]
    if abs(m) < 0.000001:
        vertex_function = lambda x: np.array([x, d])
    else:
        vertex_function = lambda x: np.array([x, c - ((d - x) / m)])

    element_to_vertex = np.zeros((n_elements, 2))
    for i in range(0, n_elements):
        element_to_vertex[i, :] = (i, i + 1)
    element_to_vertex = element_to_vertex.astype(int)

    return Mesh(vertex_function, vertex_params, element_to_vertex)

def circular_mesh(n_elements, radius, basis_fncs = None):
    # Use linear basis by default
    if basis_fncs is None:
        basis_fncs = BasisFunctions.from_degree(1)

    n_vertices = n_elements
    vertex_params = np.linspace(0, 2 * np.pi, n_vertices + 1)
    boundary_func = lambda t: radius * np.sin([np.pi / 2 - t, t])

    element_to_vertex = np.zeros((n_elements, 2))
    element_to_vertex_params = np.zeros((n_elements, 2))
    for i in range(0, n_elements):
        element_to_vertex[i, :] = (i, i + 1)
        element_to_vertex_params[i, :] = (vertex_params[i],
                                          vertex_params[i + 1])
    element_to_vertex[-1, :] = (n_elements - 1, 0)
    element_to_vertex = element_to_vertex.astype(int)

    C = Mesh(boundary_func,
            vertex_params[:-1], element_to_vertex,
            basis_fncs, element_to_vertex_params)
    return C

def combine_meshes(mesh1, mesh2):
    """
    Combine two meshes into one disconnected mesh. This function
    relies on the user to make sure that nothing weird is going on.
    for example, the meshes probably should not intersect. I'm not
    sure what would happen if they do! Also, I assume that all
    the meshes are linear (linear mapping from real to reference space).
    """
    buffer = 1.0
    max_mesh1_param = np.max(mesh1.vertex_params)
    min_mesh2_param = np.min(mesh2.vertex_params)
    num_mesh1_verts = mesh1.vertex_params.shape[0]
    vertex_params = np.append(mesh1.vertex_params,
                              mesh2.vertex_params - min_mesh2_param +
                                max_mesh1_param + buffer)

    element_to_vertex = np.vstack((mesh1.element_to_vertex,
                                  mesh2.element_to_vertex +
                                  num_mesh1_verts))

    element_to_vertex_params = \
        np.vstack((mesh1.element_to_vertex_params,
                mesh2.element_to_vertex_params -
                    min_mesh2_param +
                    max_mesh1_param + buffer))

    # Map the two sets of vertex params into two different intervals
    def boundary_func(t):
        if t < max_mesh1_param + (buffer / 2.0):
            return mesh1.boundary_fnc(t)
        else:
            return mesh2.boundary_fnc(t - buffer - max_mesh1_param +
                                        min_mesh2_param)
    return Mesh(boundary_func, vertex_params, element_to_vertex,
                None, element_to_vertex_params)

