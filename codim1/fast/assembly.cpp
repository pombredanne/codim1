#include "assembly.h"

void InteriorPoint::process_element(MappingEval& mapping, Kernel &kernel, 
                     Basis& basis, QuadratureInfo& quad_info)
{
    std::vector<std::vector<double> > integral;
    for(int i = 0; i < basis.n_fncs; i++)
    {
        integral = single_integral(mapping, kernel, one, basis, quad_info, 0, i);
        result[0] += integral[0][0];
        result[0] += integral[0][1];
        result[1] += integral[1][0];
        result[1] += integral[1][1];
    }
}

void AlignedInteriorPoint::process_element(MappingEval& mapping, Kernel &kernel, 
                     Basis& basis, QuadratureInfo& quad_info)
{
    std::vector<std::vector<double> > integral;
    for(int i = 0; i < basis.n_fncs; i++)
    {
        integral = aligned_single_integral(mapping, kernel, basis, quad_info, i);
        // I don't think there's any need to have two bases for this
        // type of integral...
        result[0] += integral[0][0];
        result[0] += integral[0][1];
        result[1] += integral[1][0];
        result[1] += integral[1][1];
    }
}
