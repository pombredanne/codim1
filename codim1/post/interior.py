from codim1.fast_lib import InteriorPoint, ConstantBasis
from codim1.assembly.shared import _choose_basis
from codim1.assembly.which_kernels import _make_which_kernels

one = ConstantBasis([1.0, 1.0])
def sgbem_interior(mesh, pt, normal, kernel_set, type):
    i = InteriorPoint()
    which_kernels = _make_which_kernels(kernel_set)
    for e_l in mesh:
        # Handle the boundary condition first
        _interior_element(i, pt, normal, type, e_l, which_kernels, "bc")
        _interior_element(i, e_l, i, "soln")

        kernel, factor = interior_which_kernels[type][e_l.bc.type]
        e_l_basis = _choose_basis(e_l.basis, kernel.soln_gradient)
        assert(kernel.test_gradient == False)
    return np.array(i.result)

def _interior_element(integrator, pt, normal, type,
                      e_l, which_kernels, bc_or_soln):
    if bc_or_soln == "bc":
        init_e_l_basis = e_l.bc.basis
        e_l_type = e_l.bc.type
    else:
        init_e_l_basis = e_l.basis
        e_l_type = which_kernels[e_l.bc.type.opposite]

    kernel, factor = which_kernels[type][e_l_type]["interior"]
    kernel.set_interior_data(pt, normal)
    if kernel is None:
        return

    e_l_basis, e_l_pt_srcs = _choose_basis(init_e_l_basis, kernel.soln_gradient)

    # Handle the integration of point sources
    for e_l_pt in e_l_pt_srcs:
        strength = [e_l_pt[0][1], e_l_pt[0][2]]
        ref_pt = e_l_pt[0][0]
        integrator.process_point_source(e_l.mapping.eval,
                                        kernel, ref_pt, strength

    if type(e_l_basis) is ZeroBasis:
        return

    # Handle the integration of basis functions
    quad_info = e_k.qs.get_point_source_quadrature(
            kernel.singularity_type, loc, e_k)
    integral = i.process_element(e_l.mapping.eval,
                                 kernel,
                                 e_l_basis,
                                 quad_inner)
