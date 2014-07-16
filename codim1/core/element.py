import numpy as np
from math import sqrt

"""
I think that integration operations should become element-wise.
The integration controller should be a "mediator" between the
basis functions, the boundary conditions, the kernel, and the mapping.

These are the four pieces of information required to integrate something.
mapping and basis are stored locally on an element.
What to integrate is handled locally by each type of integrator.

This will massively simplify most of the assembly routines.
"""

class Vertex(object):
    # Don't mess with this next_id! It will be bad. It is used for global
    # indexing of the vertices
    next_id = 1
    """Very simple class to allow avoiding some repeated math on vertices."""
    def __init__(self, loc, param = 0):
        self.id = Vertex.next_id
        Vertex.next_id += 1
        self.loc = np.array(loc)
        self.param = param
        self.connected_to = []

    def __setstate__(self, state):
        self.__dict__ = state
        Vertex.next_id = max(Vertex.next_id, self.id) + 1

    def connect_to_element(self, elem):
        if elem not in self.connected_to:
            self.connected_to.append(elem)


class MisorientationException(Exception): pass

class Element(object):
    def __init__(self, vertex1, vertex2):
        self.reinit(vertex1, vertex2)
        # An alternative to declaring these variables here would be to
        # create some way of copying the structure of a mesh and store each
        # piece in its relevant action class (dofs in DOFHandler, etc)
        self.mapping = "Undefined mapping. Apply a mapping!"
        self.basis = "Undefined basis. Apply a basis!"
        self.bc = "Undefined boundary conditions. Apply a BC!"
        self.continuous = "Undefined continuity. Set to True or False!"
        self.dofs = "Undefined dof list. Use a DOFHandler."
        self.dofs_initialized = False
        self.data = dict()

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

    def _update_left_neighbors(self):
        for e in self.neighbors_left:
            e.update_neighbors()

    def _update_right_neighbors(self):
        for e in self.neighbors_left:
            e.update_neighbors()

    def _check_for_misorientation_right(self):
        for nr in self.neighbors_right:
            if self not in nr.neighbors_left:
                raise MisorientationException()

    def _check_for_misorientation_left(self):
        for nr in self.neighbors_left:
            if self not in nr.neighbors_right:
                raise MisorientationException()

    def _check_for_misorientation(self):
        self._check_for_misorientation_left()
        self._check_for_misorientation_right()

    def set_id(self, id):
        # TODO: This id link the element back to its location in the
        # mesh. This is a bad thing. Remove this reverse dependency.
        # Might take a bit of work. Especially with respect to the
        # inter-element distance calculations done by quad strategy. But,
        # those will hopefully be unnecessary soon.
        self.id = id


def apply_to_elements(element_list, property_name,
                      value_gen, non_gen = False):
    """
    This function can be used to set a general property of some iterable set
    of elements. For example to set the the type of functional basis used on
    all the elements in a mesh, write:

    apply_to_elements(mesh, "basis",
                      BasisFunctions.from_degree(degree),
                      non_gen = True)

    In this example, non_gen = True, which means that the provided property
    value should not be copied for each element. In other words, all the
    elements will reference the same BasisFunctions object.

    If non_gen = False, the property value should be a function taking
    the element as its only input. Apply a reference to physical mapping is
    a case where this is desirable because the mapping needs to be specialized
    for each element.
    """
    if non_gen:
        value_gen_fnc = lambda e: value_gen
    else:
        value_gen_fnc = value_gen
    for e in element_list:
        e.__dict__[property_name] = value_gen_fnc(e)
