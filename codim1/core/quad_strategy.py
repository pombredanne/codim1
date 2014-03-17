import quadrature
import numpy as np

class QuadStrategy(object):
    """
    This class determines what type and order of quadrature should be used
    for each integration.
    """
    def __init__(self,
                 mesh,
                 quad_points_min,
                 quad_points_max,
                 quad_points_logr,
                 quad_points_oneoverr):
        self.mesh = mesh
        self.min_points = quad_points_min
        self.max_points = quad_points_max
        self.quad_points_logr = quad_points_logr
        self.quad_points_oneoverr = quad_points_oneoverr
        self.setup_quadrature()

    def setup_quadrature(self):
        """
        The quadrature rules can be defined once on the reference element
        and then a change of variables allows integration on any element.
        """
        self.quad_nonsingular = dict()
        for n_q in range(self.min_points - 1, self.max_points):
            self.quad_nonsingular[n_q + 1] = quadrature.QuadGauss(n_q + 1)

        self.highest_nonsingular = \
            self.quad_nonsingular[self.max_points]

        self.quad_shared_edge_left = \
            quadrature.QuadSingularTelles(self.quad_points_logr, 0.0)
        self.quad_shared_edge_right = \
            quadrature.QuadSingularTelles(self.quad_points_logr, 1.0)

        self.quad_logr = []
        self.quad_oneoverr = []
        # For each point of the outer quadrature formula, we need a different
        # singular quadrature formula, because the singularity location will
        # move.
        # The highest order nonsingular quadrature is used for the outer
        # quadrature in the case of a singular kernel.
        for singular_pt in self.highest_nonsingular.x:
            logr = quadrature.QuadSingularTelles(self.quad_points_logr,
                                                 singular_pt)
            oneoverr = quadrature.QuadOneOverR(self.quad_points_oneoverr,
                                               singular_pt,
                                               self.max_points)
            self.quad_logr.append(logr)
            self.quad_oneoverr.append(oneoverr)

    def get_simple(self):
        """Get whatever quadrature rule is used for a non singular case."""
        return self.highest_nonsingular

    def get_quadrature(self, singularity_type, k, l):
        """
        This function computes which quadrature formula should be used in
        which case.
        We use a Telles integration formula for log(r) singular cases
        and all edges
        A Piessen method is used for 1/r singularities.
        """

        # The double integration methods require one quadrature formula
        # per point of the outer quadrature. So, if we want them to all
        # be the same, just send a bunch of references to the same
        # formula.
        def _make_inner(quad):
            return [quad] * len(outer.x)

        which_nonsingular = self.choose_nonsingular(k, l)
        outer = self.quad_nonsingular[which_nonsingular]
        if k == l:
            if singularity_type == 'logr':
                inner = self.quad_logr
            elif singularity_type == 'oneoverr':
                inner = self.quad_oneoverr
        # element l is to the left of element k, but we care about the
        # quadrature from the perspective of element l, so we need to use the
        # right sided quadrature
        elif self.mesh.is_neighbor(k, l, 'left'):
            inner = _make_inner(self.quad_shared_edge_right)
        elif self.mesh.is_neighbor(k, l, 'right'):
            inner = _make_inner(self.quad_shared_edge_left)
        else:
            inner = _make_inner(outer)
        return outer, inner

    def choose_nonsingular(self, k, l):
        dist = self.mesh.element_distances[k, l]
        # Is the source k or is the source l? Maybe I should use the average
        # of the two? Find some literature on choosing quadrature formulas.
        # The choice of which element to use in computing the width is
        # irrelevant until elements get highly variable in width.
        source_width = self.mesh.element_widths[k]
        ratio = dist / source_width
        how_far = np.floor(ratio)

        points = self.max_points - how_far
        if points < self.min_points:
            points = self.min_points
        return points
