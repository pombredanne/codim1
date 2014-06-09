def _choose_basis(basis, is_gradient):
    if is_gradient:
        pt_src_info = zip(basis.point_sources, basis.point_source_dependency)
        return basis.get_gradient_basis(), pt_src_info
    return basis, []
