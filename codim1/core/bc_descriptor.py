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
two modules seems to be in order. This is done!
"""
from collections import namedtuple
BC = namedtuple("BC", "type,basis")
