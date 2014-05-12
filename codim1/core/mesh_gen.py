import numpy as np
from mesh import Mesh
from basis_funcs import BasisFunctions

def simple_line_mesh(n_elements,
        left_edge = (-1.0, 0.0), right_edge = (1.0, 0.0)):
    """
    Create a mesh consisting of a line of elements starting at -1 and
    extending to +1 in x coordinate, y = 0.
    """
    n_vertices = n_elements + 1
    x = np.linspace(left_edge[0], right_edge[0], n_vertices)
    y = np.linspace(left_edge[1], right_edge[1], n_vertices)
    vertices = np.vstack((x, y)).T

    element_to_vertex = np.zeros((n_elements, 2))
    for i in range(0, n_elements):
        element_to_vertex[i, :] = (i, i + 1)
    element_to_vertex = element_to_vertex.astype(int)

    return Mesh(vertices, element_to_vertex)

def circular_mesh(n_elements, radius):
    n_vertices = n_elements

    t = np.linspace(0, 2 * np.pi, n_vertices + 1)[:-1]
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    vertices = np.vstack((x, y)).T

    element_to_vertex = np.zeros((n_elements, 2))
    for i in range(0, n_elements - 1):
        element_to_vertex[i, :] = (i, i + 1)
    element_to_vertex[-1, :] = (n_elements - 1, 0)
    element_to_vertex = element_to_vertex.astype(int)

    return Mesh(vertices, element_to_vertex)

def combine_meshes(mesh1, mesh2, ensure_continuity = False):
    """
    Combine two meshes into one disconnected mesh. This function
    relies on the user to make sure that nothing weird is going on.
    for example, the meshes probably should not intersect. I'm not
    sure what would happen if they do! Also, I assume that all
    the meshes are linear (linear mapping from real to reference space).
    """
    buffer = 1.0
    new_vertices = np.vstack((mesh1.vertices, mesh2.vertices))
    mesh2_initial_vertex_idx = mesh1.vertices.shape[0]
    mesh2_element_to_vertex = mesh2.element_to_vertex +\
                              mesh2_initial_vertex_idx
    new_etov = np.vstack((mesh1.element_to_vertex, mesh2_element_to_vertex))

    result =  Mesh(new_vertices, new_etov)
    if ensure_continuity:
        result.condense_duplicate_vertices()
    result.parts.append(mesh1)
    result.parts.append(mesh2)
    return result

