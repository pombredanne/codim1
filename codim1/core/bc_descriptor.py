"""
Types of BCs
Traction
Displacement
Traction on a crack
Displacement Discontinuity

Ways of applying a boundary condition
Zero -- should be special since many terms do not need to be computed
Constant -- also should be special since the boundary condition can be
            extracted from the integral
Function of physical space.
Function of reference space.
Point source BC -- applying this to an element would be a nice way to get
  around the difficulties with in_element.
Apply it pointwise and interpolate in between

How to develop a unified interface for all of these?
Point sources stick out as funny and difficult -- maybe these should be a
special case. And generally avoided!

Pass in an element and a reference location?
The mapping can be used to get the physical point
Need a behind-the-scenes way of preventing the mapping from recomputing
the physical point each time, maybe just pass the physical point?
In an ideal world, we can exactly precompute all the physical points.
Maybe this is the easiest route... in such a world, the mapping functions do
not actually ever do any work during the integration routines. they simply
pop out precomputed values.

Important:
These seem very closely related to basis functions. An integration of the
two modules seems to be in order.
"""
from basis_funcs import Function
class FunctionBC(Function):
    def __init__(self, type, f, element):
        super(FunctionBC, self).__init__(f)
        self.type = type
        self.element = element

class ConstantBC(object):
    def __init__(self, type, value, element):
        self.type = type
        self.value = value
        self.element = element

    def get_value(self, x_hat):
        return self.value

class ZeroBC(ConstantBC):
    def __init__(self, type, element):
        super(ZeroBC, self).__init__(type, 0, element)
