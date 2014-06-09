
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
                }
            },
            "crack_traction":
            {
                "crack_traction":
                {
                    "matrix": (kernel_set.k_rh, -0.5),
                    "rhs": (None, 0)
                }
                # "displacement":
                # {
                #     "matrix": (kernel_set.k_tp, 1),
                #     "rhs": (kernel_set.k_rh, 1)
                # },
                # "traction":
                # {
                #     "matrix": (kernel_set.k_rh, -1),
                #     "rhs": (kernel_set.k_tp, -1)
                # }
            },
            "displacement_discontinuity":
            {

            }
        }
    return which_kernels
