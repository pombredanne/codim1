import numpy as np
from math import sqrt

class Vertex(object):
    """Very simple class to allow avoiding some repeated math on vertices."""
    def __init__(self, loc, param = 0):
        self.loc = np.array(loc)
        self.param = param
        self.connected_to = []

    def connect_to_element(self, elem):
        if elem not in self.connected_to:
            self.connected_to.append(elem)


class Element(object):
    def __init__(self, vertex1, vertex2):
        self.reinit(vertex1, vertex2)
        self.mapping = "Undefined mapping. Apply a mapping!"

    def reinit(self, vertex1, vertex2):
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.vertex1.connect_to_element(self)
        self.vertex2.connect_to_element(self)
        self.length = sqrt((self.vertex1.loc[0] - self.vertex2.loc[0]) ** 2 +
                           (self.vertex1.loc[1] - self.vertex2.loc[1]) ** 2)
        self.update_neighbors()

    def update_neighbors(self):
        """
        Checks the vertices touching this element for new neighbors.
        This is done by looking at the neighbor list for those vertices.
        """
        self.neighbors_left = []
        self.neighbors_left.extend(self.vertex1.connected_to)
        self.neighbors_left.remove(self)
        self.neighbors_right = []
        self.neighbors_right.extend(self.vertex2.connected_to)
        self.neighbors_right.remove(self)

    def set_id(self, id):
        self.id = id
