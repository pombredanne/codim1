from element import Vertex, Element

class Mesh(object):
    """
    A class for managing a one dimensional mesh within a two dimensional
    boundary element code.

    A mesh is defined by vertices and the element that connect them.
    To initialize a mesh, pass a list of vertices and a list of elements.

    Normal vectors are computed assuming that they point left of the vertex
    direction. Meaning, if \vec{r} = \vec{x}_1 - \vec{x}_0 and
    \vec{r} = (r_x, r_y) then \vec{n} = \frac{(r_x, -r_y)}{\norm{r}}. So,
    exterior domain boundaries should be traversed clockwise and
    interior domain boundaries should be traversed counterclockwise.
    """
    def __init__(self, vertices, elements):
        # Elements connect vertices and define edge properties
        self.elements = elements
        self.n_elements = len(elements)

        # Vertices contains the position of each vertex in tuple form (x, y)
        self.vertices = vertices
        self.n_vertices = len(self.vertices)

        for e_idx in range(self.n_elements):
            self.elements[e_idx].set_id(e_idx)

        # Update the adjacency lists of each element.
        self.compute_connectivity()

    def compute_connectivity(self):
        """Loop over elements and update neighbors."""
        # Determine which elements touch.
        for e in self.elements:
            e.update_neighbors()

    def condense_duplicate_vertices(self, epsilon = 1e-6):
        """
        Remove duplicate vertices and ensure continuity between their
        respective elements. This is a post-processing function on the
        already produced mesh. I don't think it will work in the
        nonlinear mesh case.

        I think it also only condenses pairs of vertices. If three vertices
        all share the same point, the problem is slightly more difficult,
        though the code should be easily adaptable.
        """
        pairs = self._find_equivalent_pairs(epsilon)
        for idx in range(len(pairs)):
            v0 = pairs[idx][0]
            v1 = pairs[idx][1]
            elements_touching = v1.connected_to
            for element in elements_touching:
                if v1 is element.vertex1:
                    element.reinit(v0, element.vertex2)
                if v1 is element.vertex2:
                    element.reinit(element.vertex2, v0)
        self.compute_connectivity()

    def _find_equivalent_pairs(self, epsilon):
        """
        To find equivalent vertex pairs, we sort the list of vertices
        by their x value first and then compare locally within the list.
        """
        # TODO: Should be extendable to find equivalence sets,
        # rather than just pairs.
        sorted_vertices = sorted(self.vertices, key = lambda v: v.loc[0])
        equivalent_pairs = []
        for (idx, v) in enumerate(sorted_vertices[:-1]):
            if abs(v.loc[0] - sorted_vertices[idx + 1].loc[0]) > epsilon:
                continue
            if abs(v.loc[1] - sorted_vertices[idx + 1].loc[1]) > epsilon:
                continue
            equivalent_pairs.append((v, sorted_vertices[idx + 1]))
        return equivalent_pairs

    def is_neighbor(self, k, l, direction = 'both'):
        """
        Return whether elements k and l are neighbors in the direction
        specified by "direction"
        For example, if element l is to the right of element k, and
        direction == 'right', this method will return true.
        """
        # Check the right side
        if (direction is 'left' or direction is 'both') \
            and self.elements[l] in self.elements[k].neighbors_left:
            return True
        # Check the left side
        if (direction is 'right' or direction is 'both') \
            and self.elements[l] in self.elements[k].neighbors_right:
            return True
        return False

    def get_neighbors(self, k, direction):
        if direction is 'left':
            return self.elements[k].neighbors_left
        if direction is 'right':
            return self.elements[k].neighbors_right
        raise Exception('When calling get_neighbors, direction should be '
                        '\'left\' or \'right\'')
