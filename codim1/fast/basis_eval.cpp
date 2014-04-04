#include "basis_eval.h"

std::vector<double> FuncEval::evaluate_vector(int element_idx, int i, 
                        double x_hat,
                        std::vector<double> x)
{
    std::vector<double> retval(2);
    retval[0] = evaluate(element_idx, i, x_hat, x, 0);
    retval[1] = evaluate(element_idx, i, x_hat, x, 1);
    return retval;
}

double FuncEval::evaluate(int element_idx, int i, double x_hat, std::vector<double> x,
                        int d) 
{
    return bp::extract<double>(function(x, d));
}


double ConstantEval::evaluate(int element_idx, int i, double x_hat,
                                std::vector<double> x,
                                int d)
{
    return value;
}

std::vector<double> ConstantEval::evaluate_vector(int element_idx, int i, 
                        double x_hat,
                        std::vector<double> x)
{
    std::vector<double> retval(2);
    retval[0] = value;
    retval[1] = value;
    return retval;
}

PolyBasisEval::PolyBasisEval(std::vector<std::vector<double> > basis)
{
    this->basis = basis;
    this->order = basis.size();
}

std::vector<double> PolyBasisEval::evaluate_vector(int element_idx, int i, 
                                double x_hat,
                                std::vector<double> x)
{
    std::vector<double> retval(2);
    retval[0] = evaluate(element_idx, i, x_hat, x, 0);
    retval[1] = retval[0];
    return retval;
}

SolutionEval::SolutionEval(PolyBasisEval &under_basis,
          std::vector<double> coeffs, 
          std::vector<std::vector<std::vector<double> > > dof_map)
{
    this->under_basis = &under_basis;
    this->coeffs = coeffs;
    this->dof_map = dof_map;
}

double SolutionEval::evaluate(int element_idx, int i, 
                              double x_hat,
                              std::vector<double> x,
                              int d)
{
    const int dof = dof_map[d][element_idx][i];
    double basis_val = under_basis->evaluate(element_idx, i, x_hat, x, d);
    return coeffs[dof] * basis_val;
}

std::vector<double> SolutionEval::evaluate_vector(int element_idx, int i, 
                        double x_hat,
                        std::vector<double> x)
{
    
    std::vector<double> retval(2);
    double basis_val = under_basis->evaluate(element_idx, i, x_hat, x, 0);
    const int dof_x = dof_map[0][element_idx][i];
    const int dof_y = dof_map[1][element_idx][i];
    retval[0] = coeffs[dof_x] * basis_val;
    retval[1] = coeffs[dof_y] * basis_val;
    return retval;
}

double PolyBasisEval::evaluate(int element_idx, int i, 
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
        a = b.evaluate(0, 1, 0.5, x, 0);
    }
    return a;
}
