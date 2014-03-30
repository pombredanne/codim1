import numpy as np

class DiscontinuousDOFHandler(object):
    """
    Maps the degrees of freedom from an element local perspective
    to a global index. Allows assembly of global matrix operators.
    The dofs on the boundary here have no support on the neighboring
    element.

    mesh is not used here, but added to the constructor to conform to
    the interface
    """
    def __init__(self, mesh, bf):
        self.dim = 2
        self.basis_fncs = bf
        self.mesh = mesh

        # element_deg + 1 degrees of freedom per element.
        # A 1st order polynomial (ax + b) has 2 coefficients
        # also one dof per dimension
        num_fncs = self.basis_fncs.num_fncs
        self.total_dofs = self.dim * self.mesh.n_elements * num_fncs

        # Making a dof_map in 1D is super simple! Its just a folded over
        # list of all the dofs
        self.dof_map = np.arange(self.total_dofs)\
                       .reshape(self.dim, self.mesh.n_elements, num_fncs)

class ContinuousDOFHandler(object):
    """
    As opposed to DiscontinuousDOFHandler, the ContinuousDOFHandler makes
    sure that adjacent elements share DOFs.
    """
    def __init__(self, mesh, bf):
        self.dim = 2
        self.basis_fncs = bf

        num_fncs = self.basis_fncs.num_fncs
        if num_fncs < 2:
            raise Exception("Continuous element degree must be at least 1.")

        self.mesh = mesh

        # Loop over elements and attach local dofs to the global dofs vector
        self.dof_map = np.empty((2, self.mesh.n_elements, num_fncs),
                                dtype = np.int)
        self.total_dofs = 0
        d = 0
        self.element_processed = [False] * mesh.n_elements
        for k in range(mesh.n_elements):
            # Handle left boundary
            nghbr_left = self.mesh.neighbors[k][0]
            if nghbr_left == -1 or not self.element_processed[nghbr_left]:
                # Far left dof.
                self.dof_map[d, k, 0] = self.total_dofs
                self.total_dofs += 1
            else:
                self.dof_map[d, k, 0] = self.dof_map[d, nghbr_left, -1]

            # Handle internal dofs.
            for i in range(1, num_fncs - 1):
                self.dof_map[d, k, i] = self.total_dofs
                self.total_dofs += 1

            # Handle right boundary
            nghbr_right = self.mesh.neighbors[k][1]
            if nghbr_right == -1 or not self.element_processed[nghbr_right]:
                # Far right dof.
                self.dof_map[d, k, -1] = self.total_dofs
                self.total_dofs += 1
            else:
                self.dof_map[d, k, -1] = self.dof_map[d, nghbr_right, 0]

            self.element_processed[k] = True

        self.dof_map[1, :, :] = self.dof_map[0, :, :] + self.total_dofs

        self.total_dofs *= 2


