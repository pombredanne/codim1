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
f = lambda x: 1 / (x ** 1.5)
a = 2
exact_value = 0.5
N = 1000

qg = QuadGauss(N, 0, 1)
t = qg.x
q = qg.w

x = a + (1 - t) / t
w = q / (t ** 2)
est = sum(w * f(x))
print est

# I could just use the QAGI routine from GNU GSL for the infinite integrands.
# In a given problem, I doubt there would ever be more than a few infinite
# elements. There would be O(n) infinite integrals to compute per infinite
# element.
# QAGI would clearly be suboptimal because it can handle arbitrary infinite
# integrals. I have a very specific type of infinite integral which decays
# like 1 / (r^a) for a = 1,2,3
