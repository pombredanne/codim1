from functools import partial
import fast_ext
from fast_ext import double_integral, single_integral, aligned_single_integral

# Let's make a whole bunch of the c++ classes picklable!
# They all have constant internal state after initialization, so
# it is simple to pickle them -- just store the initialization arguments!
class Pickleable:
    def __getinitargs__(self):
        print self.args
        return self.args

def create_init(super_type):
    def init_dynamic_type(self, *args):
        self.args = args
        super_type.__init__(self, *args)
    return init_dynamic_type

def create_pickleable_class(base_class):
    new_type = type(base_class.__name__, (base_class, Pickleable), {})
    new_type.__init__ = create_init(base_class)
    return new_type

ext_classes = ['PolyBasis', 'GradientBasis', 'SingleFunctionBasis',
               'ConstantBasis', 'ZeroBasis', 'MappingEval', 'KernelData', 'MassMatrixKernel',
               'Kernel', 'DisplacementKernel', 'TractionKernel', 'AdjointTractionKernel',
               'HypersingularKernel', 'RegularizedHypersingularKernel',
               'SemiRegularizedHypersingularKernel', 'QuadratureInfo']
for cls in ext_classes:
    globals()[cls] = create_pickleable_class(fast_ext.__dict__[cls])
