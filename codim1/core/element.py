
class Vertex(object):
    """Very simple class to allow avoiding some repeated math on vertices."""
    def __init__(self, loc):
        self.loc = loc

class Element(object):
    def __init__(self, vertex1, vertex2):
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.length = np.sqrt((self.vertex1[0] - self.vertex2[0]) ** 2 +
                              (self.vertex1[1] - self.vertex2[1]) ** 2)

