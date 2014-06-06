from codim1.fast_lib import ext_classes
import codim1.fast_lib as cfl

def test_pickle():
    for cls in ext_classes:
        cfl.__dict__[cls].__getinitargs__ == cfl.Pickleable.__getinitargs__
