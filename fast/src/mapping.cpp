#include "mapping.h"
#include <cmath>

MappingEval::MappingEval(PolyBasis& basis,
            std::vector<std::vector<double> > coeffs):
    map_basis(basis)
{
    this->coeffs = coeffs; 
}

std::vector<double>
    MappingEval::eval_function(Basis& evaluator, double x_hat)
{
    std::vector<double> val(2);
    double basis;
    for(int i = 0; i < map_basis.n_fncs; i++)
    {
        // This is assuming that the basis is the same for both dimensions
        // in the mesh. This is a reasonable assumption and is enforced
        // by the python interface layer.
        basis = evaluator.evaluate_simple(i, x_hat, 0);
        val[0] += coeffs[0][i] * basis;
        val[1] += coeffs[1][i] * basis;
    }
    return val;
}

std::vector<double>
    MappingEval::get_physical_point(double x_hat)
{
    return eval_function(map_basis, x_hat);
}

double MappingEval::get_jacobian(double x_hat)
{
    std::vector<double> deriv =
        eval_function(*map_basis.get_gradient_basis(), x_hat);
    return sqrt(pow(deriv[0], 2)  + pow(deriv[1], 2));
}

std::vector<double> MappingEval::get_normal(double x_hat)
{
    std::vector<double> deriv =
        eval_function(*map_basis.get_gradient_basis(), x_hat);
    double length = sqrt(pow(deriv[0], 2)  + pow(deriv[1], 2));
    std::vector<double> retval(2);
    retval[0] = -deriv[1] / length;
    retval[1] = deriv[0] / length;
    return retval;
}
