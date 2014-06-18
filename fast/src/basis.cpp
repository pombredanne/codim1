#include "basis.h"

using namespace codim1;
PolyBasis::PolyBasis(unsigned int degree) {
    this->degree = degree;
}

// 
// PolyBasis::PolyBasis(std::vector<std::vector<double> > basis_coeffs,
//                      std::vector<double> nodes)
// {
//     this->n_fncs = basis_coeffs.size();
//     this->basis_coeffs = basis_coeffs;
//     this->nodes = nodes;
// }
// 
// PolyBasis::PolyBasis(std::vector<std::vector<double> > basis_coeffs,
//                      std::vector<std::vector<double> > basis_derivs,
//                      std::vector<std::vector<double> > point_sources,
//                      std::vector<int> point_source_dependency,
//                      std::vector<double> nodes)
// {
//     this->n_fncs = basis_coeffs.size();
//     this->basis_coeffs = basis_coeffs;
//     this->gradient_basis.reset(new GradientBasis(basis_derivs, nodes));
//     this->point_sources = point_sources;
//     this->point_source_dependency = point_source_dependency;
//     this->nodes = nodes;
// }
// std::vector<double> PolyBasis::evaluate_vector(int i, 
//                                 double x_hat)
// {
//     std::vector<double> retval(2);
//     retval[0] = evaluate_internal(i, x_hat, 0);
//     retval[1] = retval[0];
//     return retval;
// }
// 
// double PolyBasis::evaluate_internal(int i, double x_hat, int d)
// {
//     assert(i < n_fncs);
//     double running_mult = 1.0;
//     double retval = basis_coeffs[i][n_fncs - 1];
//     for(int coeff_idx = n_fncs - 2; coeff_idx >= 0; coeff_idx--)
//     {
//         running_mult *= x_hat;
//         retval += basis_coeffs[i][coeff_idx] * running_mult;
//     }
//     return retval;
// }
