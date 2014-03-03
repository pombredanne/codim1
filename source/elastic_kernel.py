import numpy as np

class ElastostaticKernel(object):
    """
    A class to contain the elastostatic kernel calculations.

    The kernels copied in here are almost certainly wrong. I need to go over
    them carefully and compare to a trusted source. The printed versions seem
    to all get the answer slightly wrong. Maybe look in Timoshenko...
    """
    def __init__(self, shear_modulus, poisson_ratio):
        self.shear_modulus = shear_modulus
        self.poisson_ratio = poisson_ratio

    def displacement_kernel(self, r, n):
        """
        Return the displacement kernel corresponding to a
        separation of (r[0], r[1])
        This is the effect that is produced $r$ away from a unit point
        force in a full space.
        Goes by many names...
            Green's function
            Kelvin's solution
        """
        mu = self.shear_modulus
        pr = self.poisson_ratio
        dx = r[0]
        dy = r[1]
        dist_squared = dx ** 2 + dy ** 2
        dist = np.sqrt(dist_squared)

        outer_factor = 1.0 / (8.0 * np.pi * mu * (1 - pr))

        U = np.empty((2, 2))
        U[0, 0] = (-(3 - 4 * pr) * np.log(dist)) + ((dx ** 2) / dist_squared)
        U[1, 1] = -(3 - 4 * pr) * np.log(dist) + ((dy ** 2) / dist_squared)
        U[1, 0] = (dx * dy) / dist_squared
        U[0, 1] = U[1, 0]
        U *= outer_factor

        return U

    def traction_kernel(self, r, n):
        """
        Return the traction kernels.
        The normal (n) provided should be the normal to the observation point
        surface, NOT the source point surface.
        """
        mu = self.shear_modulus
        pr = self.poisson_ratio
        dx = r[0]
        dy = r[1]
        nx = n[0]
        ny = n[1]
        dist_squared = dx ** 2 + dy ** 2
        dist = np.sqrt(dist_squared)
        drdn = r.dot(n) / dist
        outer_factor = -1.0 / (4.0 * np.pi * (1 - pr) * dist)


        T = np.empty((2, 2))
        T[0, 0] = ((1 - 2 * pr) + 2 * (dx ** 2) / dist_squared) * drdn
        T[1, 1] = ((1 - 2 * pr) + 2 * (dy ** 2) / dist_squared) * drdn
        T[0, 1] = ((2 * dx * dy) / dist_squared) * drdn - \
                  (1 - 2 * pr) * ((dx / dist) * ny - (dy / dist) * nx)
        T[1, 0] = ((2 * dx * dy) / dist_squared) * drdn - \
                  (1 - 2 * pr) * ((dy / dist) * nx - (dx / dist) * ny)

        T *= outer_factor
        return T

################################################################################
# TESTS                                                                        #
################################################################################

# How to test these kernels? Bugs *will* be caught in higher level tests
# but it would be nice to be able to know where the problem is...
# Idea 1: Numerically differentiate the displacement kernel to get a
# stress kernel. Compare. Yuck!
# Idea 2:
def test_displacement():
    kernel = ElastostaticKernel(30e9, 0.25)

    x = np.linspace(-15.0, 15.0, 1000)
    y = np.zeros(1000)
    y -= 0.0
    r = np.vstack((x, y)).T

    U = np.array([kernel.displacement_kernel(r_val, 0.0) for r_val in r])

    # import matplotlib.pyplot as plt
    # plt.plot(x, U[:, 0, 1])
    # plt.show()


def test_traction():
    kernel = ElastostaticKernel(30e9, 0.25)

    pts = 150
    x = np.linspace(-5.0, 5.0, pts)
    y = np.zeros(pts)
    y -= 0.0
    r = np.vstack((x, y)).T

    T = np.array([kernel.traction_kernel(r_val, np.array([0, 1.0]))
        for r_val in r])

    # import matplotlib.pyplot as plt
    # plt.plot(x, T[:, 0, 0])
    # plt.plot(x, T[:, 0, 1] - 4.0)
    # plt.plot(x, T[:, 1, 0] - 8.0)
    # plt.plot(x, T[:, 1, 1] - 12.0)
    # plt.ylim(-15.0, 5.0)
    # plt.show()
