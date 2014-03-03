import numpy as np

class DOFHandler(object):
    """
    Maps the degrees of freedom from an element local perspective
    to a global index. Allows assembly of global matrix operators.
    """
    def __init__(self, dim, n_elements, element_deg):
        self.dim = dim
        self.n_elements = n_elements
        self.element_deg = element_deg
        # element_deg + 1 degrees of freedom per element.
        # A 1st order polynomial (ax + b) has 2 coefficients
        # also one dof per dimension
        self.total_dofs = dim * n_elements * (element_deg + 1)

        # Making a dof_map in 1D is super simple! Its just a folded over list of all
        # the dofs
        self.dof_map = np.arange(self.total_dofs)\
                       .reshape(dim, n_elements, element_deg + 1)

def test_dof_handler():
    dh = DOFHandler(2, 2, 2)
    assert(dh.dof_map[0, 0, 0] == 0)
    assert(dh.dof_map[0, 1, 1] == 4)
    assert(dh.dof_map[1, 0, 0] == 6)
