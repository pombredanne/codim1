
def _make_which_kernels(kernel_set):
    """
    A table indicating which kernel should be used for the matrix term
    and which kernel should be used for the RHS term given the type of
    boundary condition on each of the elements under consideration.

    The outer boundary condition type is the BC for the test function and
    the inner boundary condition type is the BC for the solution element.

    Use this like:
    which_kernels[e_k.bc.type][e_l.bc.type]["matrix"]
    or
    which_kernels[e_k.bc.type][e_l.bc.type]["rhs"]
    """
    which_kernels = \
        {
            "displacement":
            {
                "opposite": "traction",
                "displacement":
                {
                    "matrix": (kernel_set.k_d, 1),
                    "rhs": (kernel_set.k_t, 1),
                    "interior": (kernel_set.k_t, 1)
                },
                "traction":
                {
                    "matrix": (kernel_set.k_t, -1),
                    "rhs": (kernel_set.k_d, -1),
                    "interior": (kernel_set.k_d, -1)
                }
            },
            "traction":
            {
                "opposite": "displacement",
                "displacement":
                {
                    "matrix": (kernel_set.k_tp, 1),
                    "rhs": (kernel_set.k_rh, 1),
                    "interior": (kernel_set.k_sh, 1)
                },
                "traction":
                {
                    "matrix": (kernel_set.k_rh, -1),
                    "rhs": (kernel_set.k_tp, -1),
                    "interior": (kernel_set.k_tp, -1)
                },
                "crack_displacement":
                {
                    "matrix": (None, 0),
                    "rhs": (kernel_set.k_rh, 1.0),
                    "interior": (kernel_set.k_sh, 1.0)
                }
            },
            "crack_traction":
            {
                "opposite": "crack_displacement",
                "displacement":
                {
                    "interior": (None, 0)
                },
                "crack_traction":
                {
                    "matrix": (kernel_set.k_rh, -1.0),
                    "rhs": (None, 0),
                    "interior": (None, 0)
                },
                "crack_displacement":
                {
                    "interior": (kernel_set.k_sh, 1.0)
                }
            },
            "crack_displacement":
            {
                "opposite": "crack_traction",
                "crack_displacement":
                {
                    "matrix": (None, 0),
                    "rhs": (kernel_set.k_rh, 1.0)
                }
            }
        }
    return which_kernels
