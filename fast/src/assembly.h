#ifndef __codim1_assembly_h
#define __codim1_assembly_h
#include "basis.h"

class Kernel;
class MappingEval;
class QuadratureInfo;

class InteriorPoint
{
    public:
        InteriorPoint():
            result(2, 0.0),
            one({1.0, 1.0})
        {}

        virtual void process_element(MappingEval& mapping, Kernel &kernel, 
                    Basis& basis, QuadratureInfo& quad_info, double factor);
        virtual void process_point_source(MappingEval& mapping,
                                         Kernel &kernel,
                                         double reference_point,
                                         std::vector<double> strength,
                                         double factor);
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
                    Basis& basis, QuadratureInfo& quad_info, double factor);
};
#endif
