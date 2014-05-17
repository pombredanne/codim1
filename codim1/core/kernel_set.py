from codim1.fast_lib import *

class ElasticKernelSet(object):
    def __init__(self, shear_modulus, poisson_ratio):
        self.k_d = DisplacementKernel(shear_modulus, poisson_ratio)
        self.k_t = TractionKernel(shear_modulus, poisson_ratio)
        self.k_tp = AdjointTractionKernel(shear_modulus, poisson_ratio)
        self.k_h = HypersingularKernel(shear_modulus, poisson_ratio)
        self.k_sh =\
            SemiRegularizedHypersingularKernel(shear_modulus, poisson_ratio)
        self.k_rh =\
            RegularizedHypersingularKernel(shear_modulus, poisson_ratio)
