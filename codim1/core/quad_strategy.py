from quadrature import gauss, telles_singular, piessens, lobatto
from codim1.fast_lib import ConstantBasis,\
                            single_integral,\
                            aligned_single_integral
from mapping import distance_between_mappings
import numpy as np

"""
This whole QuadStrategy stuff is a bit of a mess.
Organize! Maybe wait until after the SGBEM stuff is implemented properly.
"""

one = ConstantBasis(np.ones(2))
def single_integral_wrapper(map_eval, kernel, basis, quad_info, which_fnc):
    """
    A wrapper so that single integral has the interface expected by the
    interior point computation functions.
    """
    return single_integral(map_eval, kernel, one,
                           basis, quad_info, 0, which_fnc)

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
        self.compute_element_distances()

    def compute_element_distances(self):
        """
        Compute the pairwise distance between all the elements. In
        2D, this is just the pairwise line segment distances. Moving to 3D,
        this shouldn't be hard if the polygons are "reasonable", but handling
        outliers may be harder. Because this distance is only used for
        selecting the quadrature strategy, I should be conservative. Using too
        many quadrature points is not as bad as using too few. Using a
        bounding box method might be highly effective.
        This might be unnecessary in a future implementation using a fast
        multipole expansion or other fast BEM method.
        """
        self.element_distances = np.zeros((self.mesh.n_elements,
                                           self.mesh.n_elements))
        for k in range(self.mesh.n_elements):
            e_k = self.mesh.elements[k]
            for l in range(self.mesh.n_elements):
                e_l = self.mesh.elements[l]
                dist = distance_between_mappings(e_k.mapping, e_l.mapping)
                self.element_distances[k, l] = dist

    def get_nonsingular_ptswts(self, n_pts):
        return gauss(n_pts)

    def setup_quadrature(self):
        """
        The quadrature rules can be defined once on the reference element
        and then a change of variables allows integration on any element.
        """
        self.quad_nonsingular = dict()
        for n_q in range(self.min_points - 1, self.max_points):
            self.quad_nonsingular[n_q + 1] = self.get_nonsingular_ptswts(n_q + 1)

        self.highest_nonsingular =  self.quad_nonsingular[self.max_points]

        self.quad_shared_edge_left=telles_singular(self.quad_points_logr, 0.0)
        self.quad_shared_edge_right=telles_singular(self.quad_points_logr, 1.0)

        self.quad_logr = []
        self.quad_oneoverr = []
        # For each point of the outer quadrature formula, we need a different
        # singular quadrature formula, because the singularity location will
        # move.
        # The highest order nonsingular quadrature is used for the outer
        # quadrature in the case of a singular kernel.
        for singular_pt in self.highest_nonsingular.x:
            logr = telles_singular(self.quad_points_logr, singular_pt)
            oneoverr = piessens(self.quad_points_oneoverr,
                                               singular_pt,
                                               self.max_points)
            self.quad_logr.append(logr)
            self.quad_oneoverr.append(oneoverr)

    def get_simple(self):
        """Get whatever quadrature rule is used for a non singular case."""
        return self.highest_nonsingular

    def get_quadrature(self, singularity_type, e_k, e_l):
        """
        This function computes which quadrature formula should be used in
        which case.
        We use a Telles integration formula for log(r) singular cases
        and all edges
        A Piessen method is used for 1/r singularities.
        """
        k = e_k.id
        l = e_l.id

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

    def get_interior_quadrature(self, e_k, pt):
        which_nonsingular = self.choose_nonsingular_interior(e_k, pt)
        interior_integrator = single_integral_wrapper
        return self.quad_nonsingular[which_nonsingular], interior_integrator

    def choose_nonsingular(self, k, l):
        dist = self.element_distances[k, l]
        # Is the source k or is the source l? Maybe I should use the average
        # of the two? Find some literature on choosing quadrature formulas.
        # The choice of which element to use in computing the width is
        # irrelevant until elements get highly variable in width.
        return self._choose_nonsingular(self.mesh.elements[k], dist)


    def choose_nonsingular_interior(self, e_k, pt):
        v1 = e_k.vertex1
        v2 = e_k.vertex2
        left_vertex_distance = v1.loc - pt
        right_vertex_distance = v2.loc - pt
        # Take the minimum of the distance from either vertex. Note that this
        # will probably overestimate the distance for a higher order mesh.
        dist = np.min([np.sqrt(np.sum(left_vertex_distance ** 2)),
                       np.sqrt(np.sum(right_vertex_distance ** 2))])
        return self._choose_nonsingular(e_k, dist)


    def _choose_nonsingular(self, e_k, dist):
        """
        Simple algorithm to choose how many points are necessary for good
        accuracy of the integration. Better algorithms are available in the
        literature. Try Sauter, Schwab 1998 or Telles 1987.
        """
        source_width = e_k.length
        ratio = dist / source_width
        how_far = np.floor(ratio)
        points = max(self.min_points, self.max_points - how_far)
        return points

    def get_point_source_quadrature(self,
                                    singularity_type,
                                    singular_pt,
                                    e_k,
                                    in_element = False,
                                    reference_loc = 0.0):
        if singularity_type == 'logr' and in_element:
            quad = quadrature.QuadSingularTelles(self.quad_points_logr,
                                                 reference_loc)
        elif singularity_type == 'oneoverr' and in_element:
            quad = quadrature.QuadOneOverR(self.quad_points_oneoverr,
                                               reference_loc,
                                               self.max_points)
        else:
            which = self.choose_nonsingular_interior(e_k, singular_pt)
            quad = self.quad_nonsingular[which]
        return quad


class GLLQuadStrategy(QuadStrategy):
    """
    This QuadStrategy simply changes the Gaussian quadrature method to a
    Gauss-Lobatto quadrature method. The points for this quadrature method
    are aligned with the interpolation nodes of a Gauss-Lobatto-Lagrange
    interpolating basis. By using an aligned set of quadrature point and
    interpolation nodes, the integration can be sped up substantially.
    """
    def __init__(self,
                 mesh,
                 n_basis_nodes,
                 quad_points_max,
                 quad_points_logr,
                 quad_points_oneoverr):
        self.n_basis_nodes = n_basis_nodes
        super(GLLQuadStrategy, self).__init__(mesh,
                                              n_basis_nodes,
                                              quad_points_max,
                                              quad_points_logr,
                                              quad_points_oneoverr)

    def get_interior_quadrature(self, e_k, pt):
        which_nonsingular = self.choose_nonsingular_interior(e_k, pt)
        interior_integrator = single_integral_wrapper
        if which_nonsingular == self.n_basis_nodes:
            interior_integrator = aligned_single_integral
        return self.quad_nonsingular[which_nonsingular], interior_integrator

    def get_nonsingular_ptswts(self, n_pts):
        return lobatto(n_pts)
