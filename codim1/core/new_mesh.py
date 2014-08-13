from copy import copy

# Is this object oriented design better than an array indexing database
# style of design? The array indexing method is probably faster and is more
# standard. However, I like the idea that mesh elements only exist if some
# other element is referencing them. I still need an overall data structure
# for the top level elements.

# Arguments for the object oriented design:
# more understandable (vertices are shared by edges which are shared by
# tris which are shared by tets)
# better data locality
# think globally act locally. more modular
# meshes are better decoupled from the mesh processing stages and can even
# be integrated with other meshes without damaging the data structures
# involved.

# Arguments for the database style:
# less complexity
# more traditional
# the reduced level of object method and attribute access might result in fast
# kernel runs

# i am inclined to choose the object oriented version because it results in
# cleaner code. Strive for the platonic ideal! But be practical.

# Barycentric coordinates makes simplices much easier
# x = sum from i = 0 to degree of b_i * vertex_i

# Refinement of simplices can be reduced to refinement of the component
# simplices plus the refinement of the overall simplex. To refine a triangle
# refine each edge and then add triangles connecting those edges. 3 of the
# new triangles will

# Refining a vertex produces itself
# Refining an edge produces a vertex and two new edges
# Refining a triangle produces three vertices and nine new edges and four new
# triangles
# Refining a tet produces
##### six new vertices (3,2,1,0 per side),
##### 24 new edges (9,7,5,3 per side
##### how many new tris?
##### how many new tets?
class WrongNumberOfSimplexComponents(object): pass

class Simplex(object):
    """
    Simplices are the simplest finite objects one can define in a given
    space. By degree they are:
    -- vertices -- degree 0
    -- edges -- degree 1
    -- triangles -- degree 2
    -- tetrahedrons(tets) -- degree 3

    A simplex has four relations:
    -- the owner is the simplex of degree n + 1
    -- the components are the simplices of degree n - 1
    -- the children are the refinement level r + 1
    -- the parent is the refinement level r - 1
    """
    def set_owner(self, owner):
        assert(self.owner is None)
        self.owner = owner

    def unique_vertices(self):
    # TODO: Raise unimplemented exception.
        pass

    def centroid(self):
        vs = self.unique_vertices()
        centroid = copy(vs[0].loc)
        for dim in range(self.dim()):
            for v in vs[1:]:
                centroid[dim] += v.loc[dim]
            centroid[dim] /= len(vs)
        return Vertex(centroid)

    def dim(self):
        return self.components[0].dim()

    def __repr__(self):
        return str(self)

class Vertex(Simplex):
    def __init__(self, loc):
        self.loc = loc
        # TODO: How to deal with a null owner
        self.owner = "vertex"

    def __str__(self):
        return "Vertex(" + str(self.loc) + ")"

    def unique_vertices(self):
        return [self]

    def dim(self):
        return len(self.loc)

    def refine(self):
        # copying the vertex is necessary here because the owner of
        # a refined vertex will be different than the owner of the unrefined
        # the owner will also be refined so the paired vertex of the edge
        # might be different
        return Vertex(copy(loc))

class Edge(Simplex):
    def __init__(self, pts):
        if len(pts) != 2:
            raise WrongNumberOfSimplexComponents("Edges contain two vertices.")
        assert(pts[0].dim() == pts[1].dim())
        self.components = pts

    def __str__(self):
        return "Edge(pt1=" + str(self.components[0]) + ", " + \
                    "pt2=" + str(self.components[1]) + ")"

    def unique_vertices(self):
        [

    def refine(self):
        # TODO: call super class method that runs the components' refine
        # methods.
        refined_comp = [
        midpt = self.centroid()
        new_edges = [
            Edge([self.components[0], midpt]),
            Edge([midpt, self.components[1]])
        ]
        return new_edges



