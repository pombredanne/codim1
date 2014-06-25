#include "basis.h"

using namespace codim1;
BasisImpl::BasisImpl(unsigned int degree):
    degree(degree)
{}

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
