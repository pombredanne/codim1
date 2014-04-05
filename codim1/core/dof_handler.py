import numpy as np

class DOFHandler(object):
    """
    This DOFHandler allows for a mixture of continuous and discontinuous
    elements. Adjacent continuous elements share dofs. Otherwise, dofs
    are not shared. Because all other basis functions go to zero at the
    boundary, matching the edge dofs is sufficient for continuity.
    """
    def __init__(self, mesh, bf, discontinuous_elements = []):
        self.discontinuous_elements = discontinuous_elements
        self.dim = 2
        self.basis_fncs = bf
        self.mesh = mesh

        num_fncs = self.basis_fncs.num_fncs
        if num_fncs < 2 and \
                len(discontinuous_elements) != self.mesh.n_elements:
            raise Exception("Continuous element degree must be at least 1.")

        # Loop over elements and attach local dofs to the global dofs vector
        self.dof_map = np.empty((2, self.mesh.n_elements, num_fncs),
                                dtype = np.int)
        self.total_dofs = 0
        d = 0
        element_processed = [False] * mesh.n_elements

        for k in range(mesh.n_elements):
            if k in discontinuous_elements:
                # If the element is discontinous, simply add some dofs
                self.process_discontinuous_element(k)
            else:
                # Otherwise, do some more complex decision making
                # as to whether
                # to relate the dof to the next door elements
                self.process_continuous_element(k, element_processed)
            element_processed[k] = True
        self.dof_map[1, :, :] = self.dof_map[0, :, :] + self.total_dofs
        self.total_dofs *= 2

    def process_discontinuous_element(self, k):
        for i in range(self.basis_fncs.num_fncs):
            self.dof_map[0, k, i] = self.total_dofs
            self.total_dofs += 1

    def process_continuous_element(self, k, is_processed):
        """
        Compute the dofs for a continuous element. The left and right sides
        should match the dofs of the neighboring element on that side.
        """
        # Handle left boundary
        nghbr_left = self.mesh.neighbors[k][0]
        if nghbr_left == -1 or \
                not is_processed[nghbr_left] or \
                nghbr_left in self.discontinuous_elements:
            # Far left dof.
            self.dof_map[0, k, 0] = self.total_dofs
            self.total_dofs += 1
        else:
            self.dof_map[0, k, 0] = self.dof_map[0, nghbr_left, -1]

        # Handle internal dofs.
        for i in range(1, self.basis_fncs.num_fncs - 1):
            self.dof_map[0, k, i] = self.total_dofs
            self.total_dofs += 1

        # Handle right boundary
        nghbr_right = self.mesh.neighbors[k][1]
        if nghbr_right == -1 or \
                not is_processed[nghbr_right] or \
                nghbr_right in self.discontinuous_elements:
            # Far right dof.
            self.dof_map[0, k, -1] = self.total_dofs
            self.total_dofs += 1
        else:
            self.dof_map[0, k, -1] = self.dof_map[0, nghbr_right, 0]
        return self.total_dofs
