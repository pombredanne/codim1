import numpy as np
def init_dofs(mesh):
    """
    Walk over the mesh and assign dofs to each element based on the
    number of basis functions.
    """
    next_dof = 0
    for e in mesh:
        n_bfs = e.basis.n_fncs
        e.dofs = np.zeros((2, n_bfs), dtype = np.int64)
        if e.continuous:
            next_dof = _process_continuous_element(e, next_dof)
        else:
            next_dof = _process_discontinuous_element(e, next_dof)
        e.dofs_initialized = True
    total_x_dofs = next_dof
    for e in mesh:
        e.dofs[1, :] = e.dofs[0, :] + total_x_dofs
    total_dofs = 2 * total_x_dofs
    mesh.total_dofs = total_dofs
    return total_dofs

def _process_discontinuous_element(e, next_dof):
    for i in range(e.basis.n_fncs):
        e.dofs[0, i] = next_dof
        next_dof += 1
    return next_dof

def _process_continuous_element(e, next_dof):
    """
    Compute the dofs for a continuous element. The left and right sides
    should match the dofs of the neighboring element on that side.
    """
    # Handle left boundary
    nghbrs_left = e.neighbors_left
    processed_from_neighbor = False
    for nl in nghbrs_left:
        if nl.dofs_initialized and nl.continuous:
            # Rightmost dof of the element to the left is dof -1.
            e.dofs[0, 0] = nl.dofs[0, -1]
            processed_from_neighbor = True
            break
    if not processed_from_neighbor:
        e.dofs[0, 0] = next_dof
        next_dof += 1

    # Handle internal dofs.
    for i in range(1, e.basis.n_fncs - 1):
        e.dofs[0, i] = next_dof
        next_dof += 1

    # Handle right boundary
    nghbrs_right = e.neighbors_right
    processed_from_neighbor = False
    for nr in nghbrs_right:
        if nr.dofs_initialized and nr.continuous:
            # Leftmost dof of the element to the right is dof 0.
            e.dofs[0, -1] = nr.dofs[0, 0]
            processed_from_neighbor = True
            break
    if not processed_from_neighbor:
        e.dofs[0, -1] = next_dof
        next_dof += 1
    return next_dof
