#include "basis.h"
#include <assert.h>

double ConstantBasis::evaluate_internal(int i, double x_hat,
                                int d)
{
    return values[d];
}

std::vector<double> ConstantBasis::evaluate_vector(int i, 
                        double x_hat)
{
    std::vector<double> retval(2);
    retval[0] = values[0];
    retval[1] = values[1];
    return retval;
}

PolyBasis::PolyBasis(std::vector<std::vector<double> > basis_coeffs,
                     std::vector<double> nodes)
{
    this->n_fncs = basis_coeffs.size();
    this->basis_coeffs = basis_coeffs;
    this->nodes = nodes;
}

PolyBasis::PolyBasis(std::vector<std::vector<double> > basis_coeffs,
                     std::vector<std::vector<double> > basis_derivs,
                     std::vector<double> nodes)
{
    this->n_fncs = basis_coeffs.size();
    this->basis_coeffs = basis_coeffs;
    this->gradient_basis.reset(new GradientBasis(basis_derivs, nodes));
    this->nodes = nodes;
}
std::vector<double> PolyBasis::evaluate_vector(int i, 
                                double x_hat)
{
    std::vector<double> retval(2);
    retval[0] = evaluate_internal(i, x_hat, 0);
    retval[1] = retval[0];
    return retval;
}

double PolyBasis::evaluate_internal(int i, 
                                double x_hat,
                                int d)
{
    assert(i < n_fncs);
    double running_mult = 1.0;
    double retval = basis_coeffs[i][n_fncs - 1];
    for(int coeff_idx = n_fncs - 2; coeff_idx >= 0; coeff_idx--)
    {
        running_mult *= x_hat;
        retval += basis_coeffs[i][coeff_idx] * running_mult;
    }
    return retval;
}

CoeffBasis::CoeffBasis(Basis& basis, 
                             std::vector<std::vector<double> > coeffs)
{
    this->n_fncs = basis.n_fncs;
    this->coeffs = coeffs;
    this->basis = &basis;
    if (this->basis->gradient_basis)
    {
        this->gradient_basis.reset(
            new CoeffBasis(*this->basis->gradient_basis, this->coeffs));
    }
}

double CoeffBasis::evaluate_internal(int i, 
                                double x_hat,
                                int d)
{

    double basis_val = this->basis->evaluate_internal(i, x_hat, d);
    return basis_val * this->coeffs[d][i];
}

std::vector<double> CoeffBasis::evaluate_vector(int i, 
                                                   double x_hat)
{
    std::vector<double> retval(2);
    retval[0] = evaluate_internal(i, x_hat, 0);
    retval[1] = evaluate_internal(i, x_hat, 1);
    return retval;
}

double basis_speed_test(std::vector<std::vector<double> > basis)
{
    std::vector<double> nodes(2, 0.0);
    PolyBasis b(basis, nodes);
    double a;
    for(int i = 0; i < 1000000; i++)
    {
        a = b.evaluate_simple(1, 0.5, 0);
    }
    return a;
}
