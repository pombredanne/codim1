import numpy as np
from quadrature import QuadGauss

# I use the transformation x = (a + (1 - t) / t) to convert an infinite
# domain of integration into a finite one. This is just an example of the
# possibilities. The specific choice of transformation is motivated by its
# use in the GNU GSL QAGI routines.

# I want to compute \int_{a}^{+\infty} f(x) dx
# Appears to work really well for this case.
# f = lambda x: 1.0 / (x ** 2)
# Lots of points required for most any other example...
f = lambda x: 1 / (x ** 2.5)
a = 2
exact_value = 0.5
N_max = 20

for N in range(2, N_max):
    qg = QuadGauss(N, 0, 1)
    t = qg.x
    q = qg.w

    x = a + (1 - t) / t
    w = q / (t ** 2)
    est = sum(w * f(x))
    print("N = " + str(N) + "   estimate: " + str(est))


# gauss_kronrod_xi = [-0.949107912342759,
#                     -0.741531185599394,
#                     -0.405845151377397,
#                     0.0,
#                     0.949107912342759,
#                     0.741531185599394,
#                     0.405845151377397,
#                     0.0]
# gauss_kronrod_wi =

# The problem with this approach is that, in certain cases, a singularity is
# introduced and in other cases, no singularity is introduced.

# At the very minimum, it seems that some sort of error test should be used
# to make sure that the integral is properly computed. Gauss-Kronrod?

# I could just use the QAGI routine from GNU GSL for the infinite integrands.
# In a given problem, I doubt there would ever be more than a few infinite
# elements. There would be O(n) infinite integrals to compute per infinite
# element.
# QAGI would clearly be suboptimal because it can handle arbitrary infinite
# integrals. I have a very specific type of infinite integral which decays
# like 1 / (r^a) for a = 1,2,3. How suboptimal would it be? I should probably
# run some tests on the time required for various integrals.

# Look at the Gauss-Rational method in the Delves, Computational Mthds for
# Integral Equations book.

# "Infinite boundary elements" by G. Beer and J.O. Watson is probably the
# most useful reference to look at.
# Read "The finite element method for infinite domains." by Babuska - might
# have some good ideas. At least check out the papers that cite that one.
# "The finite element method with nonuniform mesh sizes for unbounded domains."
# seems to have some interesting ideas about coarsening the mesh towards
# infinity, resulting in some very large elements, and a bound of the number
# of elements that's independent of the distance to which the mesh is extended.
