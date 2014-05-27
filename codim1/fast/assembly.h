#ifndef __codim1_assembly_h
#define __codim1_assembly_h
#include <boost/python/object.hpp>
#include "elastic_kernel.h"
#include "integrator.h"
#include "basis.h"

class InteriorPoint
{
    public:
        InteriorPoint():
            result(2, 0.0),
            one({1.0, 1.0})
        {}

        virtual void process_element(MappingEval& mapping, Kernel &kernel, 
                             Basis& basis, QuadratureInfo& quad_info);
        std::vector<double> result;
        ConstantBasis one;
};

class AlignedInteriorPoint: public InteriorPoint
{
    public:
        AlignedInteriorPoint():
            InteriorPoint()
        {}

        virtual void process_element(MappingEval& mapping, Kernel &kernel, 
                             Basis& basis, QuadratureInfo& quad_info);
};
#endif
