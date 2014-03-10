# cython: profile=True
def get_physical_points(mesh, element_id, reference_pt):
    vertex_list = mesh.element_to_vertex[element_id, :]
    pt1 = mesh.vertices[vertex_list[0]]
    pt2 = mesh.vertices[vertex_list[1]]
    pt2_minus_pt1 = pt2 - pt1
    physical_pts = pt1 + reference_pt * pt2_minus_pt1
    return physical_pts
