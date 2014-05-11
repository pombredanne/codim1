
class Vertex(object):
    """Very simple class to allow avoiding some repeated math on vertices."""
    def __init__(self, loc):
        self.loc = loc

class Element(object):
    def __init__(self, vertex1, vertex2):
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.length =
