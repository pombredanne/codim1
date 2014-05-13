import numpy as np
from mesh import Mesh
from element import Element, Vertex
from mapping import apply_mapping, LinearMapping
from basis_funcs import BasisFunctions

def from_vertices_and_etov(vertices, etov):
    vertex_objs = []
    for v_idx in range(vertices.shape[0]):
        vertex_objs.append(Vertex((vertices[v_idx, 0], vertices[v_idx, 1])))

    element_objs = []
    for e_idx in range(etov.shape[0]):
        v0 = vertex_objs[etov[e_idx, 0]]
        v1 = vertex_objs[etov[e_idx, 1]]
        element_objs.append(Element(v0, v1))

    m = Mesh(vertex_objs, element_objs)
    apply_mapping(m, LinearMapping)
    return m

def simple_line_mesh(n_elements,
        left_edge = (-1.0, 0.0), right_edge = (1.0, 0.0)):
    """
    Create a mesh consisting of a line of elements starting at -1 and
    extending to +1 in x coordinate, y = 0.
    """
    n_vertices = n_elements + 1
    x_list = np.linspace(left_edge[0], right_edge[0], n_vertices)
    y_list = np.linspace(left_edge[1], right_edge[1], n_vertices)
    vertices = []
    for (x, y) in zip(x_list, y_list):
        vertices.append(Vertex(np.array((x, y))))

    elements = []
    for i in range(0, n_elements):
        v0 = vertices[i]
        v1 = vertices[i + 1]
        elements.append(Element(v0, v1))

    m = Mesh(vertices, elements)
    apply_mapping(m, LinearMapping)
    return m

def circular_mesh(n_elements, radius):
    n_vertices = n_elements

    t = np.linspace(0, 2 * np.pi, n_vertices + 1)[:-1]
    x_list = radius * np.cos(t)
    y_list = radius * np.sin(t)
    vertices = []
    for (x, y) in zip(x_list, y_list):
        vertices.append(Vertex(np.array((x, y))))

    elements = []
    for i in range(0, n_elements - 1):
        v0 = vertices[i]
        v1 = vertices[i + 1]
        elements.append(Element(v0, v1))
    elements.append(Element(vertices[n_elements - 1], vertices[0]))

    m = Mesh(vertices, elements)
    apply_mapping(m, LinearMapping)
    return m

def combine_meshes(mesh1, mesh2, ensure_continuity = False):
    """
    Combine two meshes into one disconnected mesh. This function
    relies on the user to make sure that nothing weird is going on.
    for example, the meshes probably should not intersect. I'm not
    sure what would happen if they do! Also, I assume that all
    the meshes are linear (linear mapping from real to reference space).

    Note that this function destroys the internals of mesh1 and mesh2.
    Should this be fixed?

    Also, this function does not apply any mappings -- it assumes they have
    already been applied to the elements of the subordinate meshes.
    """
    vertices = mesh1.vertices
    vertices.extend(mesh2.vertices)
    elements = mesh1.elements
    elements.extend(mesh2.elements)

    result =  Mesh(vertices, elements)
    if ensure_continuity:
        result.condense_duplicate_vertices()
    return result

