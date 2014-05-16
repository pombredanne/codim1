
class BC(object):
    def __init__(self):
        pass

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


"""
