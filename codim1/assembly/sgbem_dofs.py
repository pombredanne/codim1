import numpy as np

def sgbem_dofs(mesh):
    """
    Walk over the mesh and assign dofs to each element based on the
    number of basis functions.
    """
    next_dof = 0
    for e_k in mesh:
        n_bfs = e_k.basis.n_fncs
        e_k.dofs = np.zeros((2, n_bfs), dtype = np.int32)

        next_dof = handle_left_neighbors(e_k, next_dof)

        for i in range(1, e_k.basis.n_fncs - 1):
            next_dof = set_next_dof(e_k, i, next_dof)

        next_dof = handle_right_neighbors(e_k, next_dof)

        e_k.dofs_initialized = True
    total_x_dofs = next_dof
    for e_k in mesh:
        e_k.dofs[1, :] = e_k.dofs[0, :] + total_x_dofs
    total_dofs = 2 * total_x_dofs
    mesh.total_dofs = total_dofs
    return total_dofs

def handle_left_neighbors(e_k, next_dof):
    return handle_neighbors(e_k, 0,
                            e_k.neighbors_left, next_dof)

def handle_right_neighbors(e_k, next_dof):
    return handle_neighbors(e_k, e_k.basis.n_fncs - 1,
                            e_k.neighbors_right, next_dof)

def set_next_dof(e_k, local_dof, next):
    e_k.dofs[0, local_dof] = next
    return next + 1

def handle_neighbors(e_k, local_dof, nghbrs, next_dof):
    # Logic:
    # If this element is a displacement element (traction boundary condition)
    # , then it should be continuous
    # with any neighboring displacement element unless a neighboring element
    # is a displacement_discontinuity element
    # A traction element (displacement boundary condition should not be
    # continuous with any neighbor
    any_disc = any([e.bc.type == "crack_displacement"
                    for e in ([e_k] + nghbrs)])
    if any_disc:
        return set_next_dof(e_k, local_dof, next_dof)

    if e_k.bc.type == "displacement" or\
       e_k.bc.type == "crack_displacement":
        any_disp = any([e.bc.type == "traction"
                        for e in ([e_k] + nghbrs)])
        if any_disp:
            return set_next_dof(e_k, local_dof, next_dof)

    # if disp_or_disc:
    #     return set_next_dof(e, local_dof, next_dof)

    # If we get here, the node is meant to be continuous
    # Check if any of the neighboring nodes are initialized. If they
    # are, use the same dof idx.
    for n in nghbrs:
        if n.dofs_initialized:
            nghbr_dof = find_neighbor_dof(e_k, n)
            e_k.dofs[0, local_dof] = nghbr_dof
            return next_dof

    # If we get here, no neighbors have set their dofs yet.
    return set_next_dof(e_k, local_dof, next_dof)

def find_neighbor_dof(e_k, n):
    for nghbr in n.neighbors_left:
        if nghbr is e_k:
            return n.dofs[0, 0]
    for nghbr in n.neighbors_right:
        if nghbr is e_k:
            highest_bf = n.basis.n_fncs - 1
            return n.dofs[0, highest_bf]
    assert(False)
