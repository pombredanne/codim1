#include "basis_eval.h"

std::vector<double> FuncEval::evaluate_vector(int i, 
                        double x_hat,
                        std::vector<double> x)
{
    std::vector<double> retval(2);
    retval[0] = evaluate(i, x_hat, x, 0);
    retval[1] = evaluate(i, x_hat, x, 1);
    return retval;
}

double FuncEval::evaluate(int i, double x_hat, std::vector<double> x,
                        int d) 
{
    return bp::extract<double>(function(x, d));
}


double ConstantEval::evaluate(int i, double x_hat,
                                std::vector<double> x,
                                int d)
{
    return values[d];
}

std::vector<double> ConstantEval::evaluate_vector(int i, 
                        double x_hat,
                        std::vector<double> x)
{
    std::vector<double> retval(2);
    retval[0] = values[0];
    retval[1] = values[1];
    return retval;
}

PolyBasisEval::PolyBasisEval(std::vector<std::vector<double> > basis)
{
    this->basis = basis;
    this->order = basis.size();
}

std::vector<double> PolyBasisEval::evaluate_vector(int i, 
                                double x_hat,
                                std::vector<double> x)
{
    std::vector<double> retval(2);
    retval[0] = evaluate(i, x_hat, x, 0);
    retval[1] = retval[0];
    return retval;
}

double PolyBasisEval::evaluate(int i, 
                                double x_hat,
                                std::vector<double> x,
                                int d)
{
    double running_mult = 1.0;
    double retval = basis[i][order - 1];
    for(int coeff_idx = order - 2; coeff_idx >= 0; coeff_idx--)
    {
        running_mult *= x_hat;
        retval += basis[i][coeff_idx] * running_mult;
    }
    return retval;
}

double basis_speed_test(std::vector<std::vector<double> > basis)
{
    PolyBasisEval b(basis);
    double a;
    std::vector<double> x(2);
    x[0] = 0.5;
    x[1] = 0.0;
    for(int i = 0; i < 1000000; i++)
    {
        a = b.evaluate(1, 0.5, x, 0);
    }
    return a;
}
