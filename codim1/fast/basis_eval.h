#ifndef __codim1_basis_eval_h
#define __codim1_basis_eval_h
#include <vector>

// Basis evaluation could be sped up by manually entering
// the functions for all reasonable degrees.
class BasisEval
{
    public:
        BasisEval(std::vector<std::vector<double> > basis);

        /* Evaluate a basis function at a point.
         */
        double evaluate(int i, double x);

        std::vector<std::vector<double> > basis;
        int order;
};

BasisEval::BasisEval(std::vector<std::vector<double> > basis)
{
    this->basis = basis;
    this->order = basis.size();
}

double BasisEval::evaluate(int i, double x)
{
    double running_mult = 1.0;
    double retval = basis[i][order - 1];
    for(int coeff_idx = order - 2; coeff_idx >= 0; coeff_idx--)
    {
        running_mult *= x;
        retval += basis[i][coeff_idx] * running_mult;
    }
    return retval;
}

//A simple test to determine how much overhead the python function calling
//costs
void basis_speed_test(std::vector<std::vector<double> > basis)
{
    BasisEval b(basis);
    for(int i = 0; i < 1000000; i++)
    {
        double a = b.evaluate(1, 0.5);
    }
}
#endif
