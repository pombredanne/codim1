from quadrature import gauss, telles_singular, piessens, lobatto,\
                       telles_quasi_singular
from codim1.fast_lib import ConstantBasis,\
                            single_integral,\
                            aligned_single_integral
from mapping import distance_between_mappings
import numpy as np
from codim1.core.segment_distance import point_segment_distance
from math import floor, ceil, exp, log

'''
This whole QuadStrategy stuff is a bit of a mess.
Organize! Maybe wait until after the SGBEM stuff is implemented properly.
'''

one = ConstantBasis(np.ones(2))
def single_integral_wrapper(map_eval, kernel, basis, quad_info, which_fnc):
    '''
    A wrapper so that single integral has the interface expected by the
    interior point computation functions.
    '''
    return single_integral(map_eval, kernel, one,
                           basis, quad_info, 0, which_fnc)

class QuadStrategy(object):
    '''
    This class determines what type and order of quadrature should be used
    for each integration.
    '''
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
        '''
        Compute the pairwise distance between all the elements. In
        2D, this is just the pairwise line segment distances. Moving to 3D,
        this shouldn't be hard if the polygons are 'reasonable', but handling
        outliers may be harder. Because this distance is only used for
        selecting the quadrature strategy, I should be conservative. Using too
        many quadrature points is not as bad as using too few. Using a
        bounding box method might be highly effective.
        This might be unnecessary in a future implementation using a fast
        multipole expansion or other fast BEM method.
        '''
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
        '''
        The quadrature rules can be defined once on the reference element
        and then a change of variables allows integration on any element.
        '''
        self.quad_nonsingular = dict()
        for n_q in range(self.min_points - 1, self.max_points):
            self.quad_nonsingular[n_q + 1] = \
                self.get_nonsingular_ptswts(n_q + 1)

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
        '''Get whatever quadrature rule is used for a non singular case.'''
        return self.highest_nonsingular

    def get_nonsingular_minpts(self):
        return self.quad_nonsingular[self.min_points]

    def get_quadrature(self, singularity_type, e_k, e_l):
        '''
        This function computes which quadrature formula should be used in
        which case.
        We use a Telles integration formula for log(r) singular cases
        and all edges
        A Piessen method is used for 1/r singularities.
        '''
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
        return self.quad_nonsingular[which_nonsingular]

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
        '''
        Simple algorithm to choose how many points are necessary for good
        accuracy of the integration. Better algorithms are available in the
        literature. Try Sauter, Schwab 1998 or Telles 1987.
        '''
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
            quad = telles_singular(self.quad_points_logr,
                                                 reference_loc)
        elif singularity_type == 'oneoverr' and in_element:
            quad = piessens(self.quad_points_oneoverr,
                                               reference_loc,
                                               self.max_points)
        else:
            quad = self.get_interior_quadrature(e_k, singular_pt)
        return quad


class GLLQuadStrategy(QuadStrategy):
    '''
    This QuadStrategy simply changes the Gaussian quadrature method to a
    Gauss-Lobatto quadrature method. The points for this quadrature method
    are aligned with the interpolation nodes of a Gauss-Lobatto-Lagrange
    interpolating basis. By using an aligned set of quadrature point and
    interpolation nodes, the integration can be sped up substantially.

    WARNING:
    This QuadStrategy also assumes that the integration for an interior
    point is non-singular and can be well approximated by a Gauss-Lobatto
    quadrature rule with a number of points equal to the number of nodes
    of the basis. This may not be true for interior points near the boundary.
    '''
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
        return self.quad_nonsingular[self.n_basis_nodes]

    def get_nonsingular_ptswts(self, n_pts):
        return lobatto(n_pts)


class AdaptiveInteriorQuad(QuadStrategy):
    def __init__(self,
                 step_size,
                 min_points,
                 unit_points,
                 max_points):
        self.step_size = int(step_size)
        self.min_points = int(min_points)
        self.unit_points = int(unit_points)
        self.max_points = int(max_points)
        if (self.unit_points - self.min_points) % self.step_size != 0:
            raise Exception('Point range not divisible by step size.')
        if (self.max_points - self.min_points) % self.step_size != 0:
            raise Exception('Point range not divisible by step size.')
        N = np.arange(self.min_points,
                      self.max_points + self.step_size,
                      self.step_size)
        self.quad_rules = {n: gauss(n) for n in N}

    def get_interior_quadrature(self, e_k, pt):
        l = e_k.length
        d = point_segment_distance(pt[0], pt[1],
                                   e_k.vertex1.loc[0],
                                   e_k.vertex1.loc[1],
                                   e_k.vertex2.loc[0],
                                   e_k.vertex2.loc[1])
        ratio = d / l
        if ratio < 1:
            ratio = -l / d

        int_ratio = int(floor(ratio))
        points = self.unit_points - int_ratio * self.step_size
        points = max(min(points, self.max_points), self.min_points)
        # print('Using: ' + str(points))
        # print l, d, points
        return self.quad_rules[points]

class AdaptiveInteriorQuad2(QuadStrategy):
    '''
    Experimentation. Assumes exponential form for the error:
    E = L * A * exp(h * q)
    '''
    def __init__(self, min_order, error_ratio):
        # error_ratio is voodoo parameter
        self.min_order = min_order
        self.unit_points = min_order * 4
        self.error_ratio = error_ratio
        self.quad_rules = dict()

    def grab_quad_rule(self, N):
        if N not in self.quad_rules:
            self.quad_rules[N] = gauss(N)
        return self.quad_rules[N]

    def get_dist(self, e_k, pt):
        return point_segment_distance(pt[0], pt[1],
                                   e_k.vertex1.loc[0],
                                   e_k.vertex1.loc[1],
                                   e_k.vertex2.loc[0],
                                   e_k.vertex2.loc[1])

    def get_interior_quadrature(self, e_k, pt):
        l = e_k.length
        d = self.get_dist(e_k, pt)
        order = log(self.error_ratio) / (d / l)
        # print "Estimate: " + str(order)
        order = int(ceil(max(self.min_order, order)))
        if order > 1000:
            order = 1000
        return self.grab_quad_rule(order)



class TellesQuadStrategy(QuadStrategy):
    '''
    Use a Telles quadrature method to compute interior point integrals.
    Just a warning.
    This quadrature strategy can only be used for interior points.
    '''
    #TODO: I should refactor out a difference between 'interior point'
    # quad strategies and boundary quad strategies
    def __init__(self,
                 n_points):
        self.n_points = n_points
        self.quad_points_oneoverr = n_points
        self.max_points = n_points

    def get_interior_quadrature(self, e_k, pt):
        D, x0 = telles_distance(pt[0], pt[1],
                        e_k.vertex1.loc[0],
                        e_k.vertex1.loc[1],
                        e_k.vertex2.loc[0],
                        e_k.vertex2.loc[1])
        try:
            return telles_quasi_singular(self.n_points, x0, D)
        except:
            return telles_quasi_singular(self.n_points + 1, x0, D)


import math
def telles_distance(px, py, x1, y1, x2, y2):
    '''
    This is just a copy of segment_distance.point_segment_distance
    that also returns the nearest point
    '''
    dx = x2 - x1
    dy = y2 - y1
    # Calculate the t that minimizes the distance.
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

    # See if this represents one of the segment's
    # end points or a point in the middle.
    if t < 0:
        dx = px - x1
        dy = py - y1
        x0 = 0.0
    elif t > 1:
        dx = px - x2
        dy = py - y2
        x0 = 1.0
    else:
        x0 = t
        near_x = x1 + t * dx
        near_y = y1 + t * dy
        dx = px - near_x
        dy = py - near_y

    return math.hypot(dx, dy), x0

