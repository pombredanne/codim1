#ifndef __codim1_integration_h
#define __codim1_integration_h
#include <vector>

class MappingEval;
class Kernel;
class Basis;

struct QuadratureInfo
{
    QuadratureInfo(double x0,
                   std::vector<double> x,
                   std::vector<double> w)
    {
        this->x0 = x0;
        this->x = x;
        this->w = w;
    }
    double x0;
    std::vector<double> x;
    std::vector<double> w;
};
        
std::vector<std::vector<double> >
double_integral(MappingEval& k_mesh_eval, 
                MappingEval& l_mesh_eval,  
                Kernel& kernel, 
                Basis& k_basis_eval,
                Basis& l_basis_eval,
                QuadratureInfo& k_quadrature,
                std::vector<QuadratureInfo> l_quadrature,
                int test_basis_idx, int soln_basis_idx);
/*
 * Performs a single integral over one element. The operations are all 
 * almost identical to those in the double_integral method. Thus, read 
 * through that method for details (by some arguments, duplication of 
 * comments just creates more "code" to maintain)
 *
 * A key difference with single_integral is that the kernel function here
 * is expected to just be a standard function with a location 
 * parameter. 
 * The double integral function takes a kernel class object with 
 * a call method taking a separation input. K(x) vs. K(x - y)
 */
std::vector<std::vector<double> >
single_integral(MappingEval& k_mesh_eval, 
                Kernel& kernel, 
                Basis& i_basis_eval,
                Basis& j_basis_eval,
                QuadratureInfo& quadrature,
                int test_basis_idx, int soln_basis_idx);
#endif
