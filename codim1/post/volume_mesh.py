from math import ceil
import numpy as np
from codim1.core import Vertex, Element, PolynomialMapping
from collections import namedtuple
import meshpy.triangle as triangle
from copy import copy
from matplotlib import pyplot as plt
import matplotlib.tri as tri
from scipy.sparse import dok_matrix, csgraph

Edge = namedtuple('Edge', 'v_indices, is_boundary, meshpy_idx')

class VolumeMesh(object):
    # viewing_region should be like ((x_min, x_max), (y_min, y_max)
    def __init__(self,
                 bem_mesh,
                 viewing_region,
                 refine_length = -1,
                 refine_area = 1e9):
        self.mesh = bem_mesh
        self.region = viewing_region
        self.refine_length = refine_length
        self.refine_area = refine_area

        self.elements = copy(self.mesh.elements)
        self.add_boundaries()

        # Build the meshpy structures
        self.v_mapping = dict()
        self.marker_to_e = dict()
        self.meshpy_es = []
        self.meshpy_vs = []
        self.meshpy_markers = []
        self.es = []
        self.vs = []
        self.collect()

        # Calculate the meshpy triangulation
        self.meshpy()

        # Separate the disjoint subregions
        self.calc_subregions()
        self.identify_regions()

    def min_x(self): return self.region[0][0]
    def min_y(self): return self.region[1][0]
    def max_x(self): return self.region[0][1]
    def max_y(self): return self.region[1][1]

    # Need to add and subtract 2 so that the values 0 and 1 are not used for
    # boundary markers. These are special and reserved by triangle.
    def marker_from_e_idx(self, e_idx):
        return e_idx + 2

    def marker_to_e_idx(self, e_idx):
        return e_idx - 2

    def collect(self):
        """
        Collect vertices and facets for building an interior mesh from the
        out in.
        """
        for e in self.elements:
            factor = self.refine_factor(e.length)
            added_verts = self.check_add_vertices(e)
            if added_verts is None:
                continue
            self.add_edges_from_e(e, factor)

    def refine_factor(self, e_len):
        # Refine factor is 1 if self.refine_length < e_len or the ratio
        # other (integral!)
        return max(1, ceil(e_len / self.refine_length))

    def add_edges_from_e(self, e, refine_factor):
        # Evenly map the refined vertices along the high order mappings.
        vs_x_hat = np.linspace(0.0, 1.0, refine_factor + 1)
        vs_x = [e.mapping.get_physical_point(x_hat) for x_hat in vs_x_hat]

        # Add points in the case of refine_factor > 1
        vs = [e.vertex1]
        # # Create "Vertex" objects in order to provide an id for each vertex.
        for v_x in vs_x[1:-1]:
            new_v = Vertex(v_x)
            vs.append(new_v)
            other_vert = self.v_mapping.get(new_v.id, None)
            assert(other_vert is None)
            self.add_vertex(new_v)
        vs.append(e.vertex2)

        for i, v in enumerate(vs[:-1]):
            self.add_edge_from_indices([v.id, vs[i + 1].id], e)

    def add_edge_from_indices(self, v_indices, e):
        # Add a volumetric mesh edge from two vertices.
        new_e_idx = len(self.es)
        meshpy_indices = [self.v_mapping[v_id] for v_id in v_indices]
        self.es.append(Edge(meshpy_indices, True, len(self.meshpy_es)))
        self.meshpy_es.append(meshpy_indices)
        self.meshpy_markers.append(self.marker_from_e_idx(new_e_idx))
        self.marker_to_e[new_e_idx] = e

    def check_add_vertices(self, e):
        # If either of the vertices is in the viewing area, we want the edge
        # If we only take edges that are fully in the viewing area, then
        # intersections with the boundaries will be incomplete.
        either_in = self.in_view(e.vertex1.loc) or self.in_view(e.vertex2.loc)
        if not either_in:
            return None

        vs = [e.vertex1, e.vertex2]
        # Add vertices in case they haven't been added yet (vertices are
        # shared between elements)
        for v in vs:
            if not self.v_mapping.get(v.id, None) is None:
                continue
            self.add_vertex(v)
        return vs

    def add_vertex(self, v):
        #
        new_v_idx = len(self.vs)
        self.vs.append(v)
        self.v_mapping[v.id] = new_v_idx
        self.meshpy_vs.append(v.loc)

    def in_view(self, x):
        return self.min_x() <= x[0] <= self.max_x()\
           and self.min_y() <= x[1] <= self.max_y()

    def create_view_edge(self, e):
        e.mapping = PolynomialMapping(e)
        self.elements.append(e)

    def add_boundaries(self):
        # Add the rectangular outer boundary of the viewing area.
        lower_left = Vertex(np.array((self.min_x(), self.min_y())))
        upper_left = Vertex(np.array((self.min_x(), self.max_y())))
        upper_right = Vertex(np.array((self.max_x(), self.max_y())))
        lower_right = Vertex(np.array((self.max_x(), self.min_y())))
        self.create_view_edge(Element(lower_left, upper_left))
        self.create_view_edge(Element(upper_left, upper_right))
        self.create_view_edge(Element(upper_right, lower_right))
        self.create_view_edge(Element(lower_right, lower_left))

    def meshpy(self):
        # Call meshpy and create the delaunay triangulation.
        def refine_func(vertices, area):
            return area > self.refine_area
        xy_scaling = (self.max_y() - self.min_y()) / \
                      float(self.max_x() - self.min_x())
        info = triangle.MeshInfo()
        # Enter triangulation coordinates (so that delaunay angles are
        # reasonable) by multiplying by xy_scaling
        internal_points = map(lambda x: (x[0] * xy_scaling, x[1]),
                                   copy(self.meshpy_vs))
        info.set_points(internal_points)
        info.set_facets(self.meshpy_es, facet_markers = self.meshpy_markers)

        # Don't allow steiner points. These are points added on the boundary
        # of the domain. I need to keep track of these points, so I prevent
        # their addition. I don't think is strictly necessary, but it makes
        # the code easier for now...
        mesh = triangle.build(info,
                              refinement_func = refine_func,
                              generate_faces = True,
                              allow_boundary_steiner = False,
                              allow_volume_steiner = False)
        self.meshpy = mesh

        self.meshpy_pts = np.array(mesh.points)
        # Exit triangulation coordinates
        self.meshpy_pts[:, 0] /= xy_scaling
        self.meshpy_tris = np.array(mesh.elements, dtype = np.int)

    def viz_vertex_labels(self):
        for i in range(self.meshpy_pts.shape[0]):
            x = self.meshpy_pts[i, 0]
            y = self.meshpy_pts[i, 1]
            label_x_loc = 25
            label_y_loc = 25
            plt.annotate(self.components[i], xy = (x, y),
                         xytext = (label_x_loc, label_y_loc),
                         textcoords = 'offset points',
                         ha = 'right',
                         va = 'bottom',
                         bbox = dict(boxstyle = 'round, pad=0.5',
                                     fc = 'yellow',
                                     alpha = 0.5),
                         arrowprops = dict(arrowstyle = '->',
                                           connectionstyle = 'arc3,rad=0'))

    def viz_mesh(self, selected = False):
        plot_tris = self.meshpy_tris
        if selected:
            plot_tris = self.selected_tris
        plt.triplot(self.meshpy_pts[:, 0], self.meshpy_pts[:, 1], plot_tris)
        # for e in self.es:
        #     pt1 = self.vs[e.v_indices[0]].loc
        #     pt2 = self.vs[e.v_indices[1]].loc
        #     plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'k-', linewidth = 6)
        # self.viz_vertex_labels()
        # for r in self.regions:
        #     loc = self.region_label_loc(r)
        #     plt.text(loc[0], loc[1], r, fontsize = 24, bbox=dict(facecolor='red', alpha=0.5))
        plt.show(block = False)

    def in_component(self, tri, comp):
        for i in range(len(tri)):
            p_comp = self.components[tri[i]]
            if p_comp == comp:
                return True
        return False

    def region_label_loc(self, r):
        # Here, I just use the location of the first vertex.
        # I should use some median or mean location. It'd be a bit nicer.
        first_loc = self.components.index(r)
        return self.meshpy_pts[first_loc, :]

    def on_boundary(self, f):
        # Boundary markers are all greater than 2.
        for i in range(len(f)):
            marker = self.meshpy.point_markers[f[i]]
            if marker >= 2:
                return True
        return False

    def calc_subregions(self):
        # I calculate the connected components of the viewing region using
        # a graph theoretic approach. This is straightforward since we have
        # edges already. The edges are disconnected at boundaries so that the
        # original boundary mesh disects the viewing region into many areas.
        # This way, we can specify that only the area beneath the surface of
        # the earth should be computed and displayed.
        n_pts = self.meshpy_pts.shape[0]
        connectivity = dok_matrix((n_pts, n_pts))
        for f_idx, f in enumerate(self.meshpy.faces):
            if self.on_boundary(f):
                continue
            connectivity[f[0], f[1]] = 1
        # Connected components are computed using a matrix-based approach in
        # scipy.
        self.n_components, self.components =\
            csgraph.connected_components(connectivity,
                    directed = False,
                    return_labels = True)

            def identify_regions(self):
                self.components = list(self.components)
        self.regions = []
        for r in self.components:
            if self.components.count(r) <= 1:
                continue
            if r in self.regions:
                continue
            self.regions.append(r)
        # TODO: I have a bunch of regions. These are numbered in some unknown
        # fashion. I need to replace all the regions that only have one
        # member with a -1 and then number the remaining regions in ascending
        # order.
        # min_region = min(self.regions)
        # self.regions = [r - min_region for r in self.regions]
        # self.components = [map(lambda c: c - min_region, self.components)

    def choose_subregion(self, which_component):
        if not (0 <= which_component <= self.n_components):
            raise Exception("for choose_subregion, which_component must be" +
                    " a valid component of the interior triangulation")
        self.selected_tris = [t for t in self.meshpy_tris
                              if self.in_component(t, which_component)]
