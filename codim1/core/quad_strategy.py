import quadrature

class QuadStrategy(object):
    """
    This class determines what type and order of quadrature should be used
    for each integration.
    """
    def __init__(self,
                 mesh,
                 quad_points_nonsingular,
                 quad_points_logr,
                 quad_points_oneoverr):

        self.mesh = mesh
        self.quad_points_nonsingular = quad_points_nonsingular
        self.quad_points_logr = quad_points_logr
        self.quad_points_oneoverr = quad_points_oneoverr
        self.setup_quadrature()

    def setup_quadrature(self):
        """
        The quadrature rules can be defined once on the reference element
        and then a change of variables allows integration on any element.
        """
        self.quad_nonsingular = quadrature.QuadGauss(
                self.quad_points_nonsingular)
        self.quad_logr = []
        self.quad_oneoverr = []
        self.quad_shared_edge_left = \
            quadrature.QuadSingularTelles(self.quad_points_logr, 0.0)
        self.quad_shared_edge_right = \
            quadrature.QuadSingularTelles(self.quad_points_logr, 1.0)
        for singular_pt in self.quad_nonsingular.x:
            logr = quadrature.QuadSingularTelles(self.quad_points_logr,
                                                 singular_pt)
            oneoverr = quadrature.QuadOneOverR(self.quad_points_oneoverr,
                                               singular_pt,
                                               self.quad_points_nonsingular)
            self.quad_logr.append(logr)
            self.quad_oneoverr.append(oneoverr)

    def get_simple(self):
        return self.quad_nonsingular

    def get_quadrature(self, k, l):
        """
        This function computes which quadrature formula should be used in
        which case.
        We use a Telles integration formula for log(r) singular cases
        and all edges
        A Piessen method is used for 1/r singularities.
        """
        def make_inner(quad):
            return [quad] * len(self.quad_nonsingular.x)
        if k == l:
            G_quad = self.quad_logr
            H_quad = self.quad_oneoverr
        # element l is to the left of element k, but we care about the
        # quadrature from the perspective of element l, so we need to use the
        # right sided quadrature
        elif self.mesh.is_neighbor(k, l, 'left'):
            G_quad = make_inner(self.quad_shared_edge_right)
            H_quad = make_inner(self.quad_shared_edge_right)
        elif self.mesh.is_neighbor(k, l, 'right'):
            G_quad = make_inner(self.quad_shared_edge_left)
            H_quad = make_inner(self.quad_shared_edge_left)
        else:
            G_quad = make_inner(self.quad_nonsingular)
            H_quad = make_inner(self.quad_nonsingular)
        return (self.quad_nonsingular, G_quad), (self.quad_nonsingular, H_quad)
