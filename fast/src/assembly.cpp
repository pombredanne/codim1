#include "assembly.h"
#include "mapping.h"
#include "elastic_kernel.h"
#include "integrator.h"
#include "basis.h"

void InteriorPoint::process_element(MappingEval& mapping,
                                    Kernel &kernel, 
                                    Basis& basis,
                                    QuadratureInfo& quad_info,
                                    double factor)
{
    std::vector<std::vector<double> > integral;
    for(int i = 0; i < basis.n_fncs; i++)
    {
        integral = single_integral(mapping, kernel, one, basis, quad_info, 0, i);
        result[0] += factor * integral[0][0];
        result[0] += factor * integral[0][1];
        result[1] += factor * integral[1][0];
        result[1] += factor * integral[1][1];
    }
}

void InteriorPoint::process_point_source(MappingEval& mapping,
                                         Kernel &kernel,
                                         double reference_point,
                                         std::vector<double> strength,
                                         double factor)
{
    std::vector<double> normal = mapping.get_normal(reference_point);
    std::vector<double> phys_pt = mapping.get_physical_point(reference_point);
    KernelData kd = kernel.get_interior_integral_data(phys_pt, normal);
    result[0] += factor * kernel.call(kd, 0, 0);
    result[0] += factor * kernel.call(kd, 0, 1);
    result[1] += factor * kernel.call(kd, 1, 0);
    result[1] += factor * kernel.call(kd, 1, 1);
}

void AlignedInteriorPoint::process_element(MappingEval& mapping, 
                                           Kernel &kernel, 
                                           Basis& basis,
                                           QuadratureInfo& quad_info,
                                           double factor)
{
    std::vector<std::vector<double> > integral;
    for(int i = 0; i < basis.n_fncs; i++)
    {
        integral = aligned_single_integral(mapping, kernel, basis, quad_info, i);
        // I don't think there's any need to have two bases for this
        // type of integral...
        result[0] += factor * integral[0][0];
        result[0] += factor * integral[0][1];
        result[1] += factor * integral[1][0];
        result[1] += factor * integral[1][1];
    }
}
