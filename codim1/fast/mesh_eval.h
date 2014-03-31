#ifndef __codim1_mesh_eval_h
#define __codim1_mesh_eval_h

#include "basis_eval.h"
#include <vector>

/*
 * Performs fast evaluations for the meshing functions. Returns jacobians
 * physical points, normals.
 */
class MeshEval
{
    public:
        MeshEval(std::vector<std::vector<double> > mesh_basis,
            std::vector<std::vector<double> > mesh_derivs,
            std::vector<std::vector<std::vector<double> > > mesh_coeffs);
        std::vector<double> eval_function(BasisEval &evaluator,
                int element_idx, double x_hat);
        std::vector<double>
            get_physical_point(int element_idx, double x_hat);
        double get_jacobian(int element_idx, double x_hat);
        std::vector<double> get_normal(int element_idx,
                double x_hat);
        BasisEval basis_eval;
        BasisEval deriv_eval;
        std::vector<std::vector<std::vector<double> > > coeffs;
};

MeshEval::MeshEval(std::vector<std::vector<double> > mesh_basis,
            std::vector<std::vector<double> > mesh_derivs,
            std::vector<std::vector<std::vector<double> > > mesh_coeffs):
    basis_eval(mesh_basis),
    deriv_eval(mesh_derivs)
{
    this->coeffs = mesh_coeffs; 
}

std::vector<double>
    MeshEval::eval_function(BasisEval& evaluator, 
            int element_idx, double x_hat)
{
    std::vector<double> val(2);
    double basis;
    for(int i = 0; i < basis_eval.order; i++)
    {
        basis = evaluator.evaluate(i, x_hat);
        val[0] += coeffs[0][element_idx][i] * basis;
        val[1] += coeffs[1][element_idx][i] * basis;
    }
    return val;
}

std::vector<double>
    MeshEval::get_physical_point(int element_idx, double x_hat)
{
    return eval_function(basis_eval, element_idx, x_hat);
}

double MeshEval::get_jacobian(int element_idx, double x_hat)
{
    std::vector<double> deriv =
        eval_function(deriv_eval, element_idx, x_hat);
    return sqrt(pow(deriv[0], 2)  + pow(deriv[1], 2));
}

std::vector<double> MeshEval::get_normal(int element_idx, double x_hat)
{
    std::vector<double> deriv =
        eval_function(deriv_eval, element_idx, x_hat);
    double length = sqrt(pow(deriv[0], 2)  + pow(deriv[1], 2));
    std::vector<double> retval(2);
    retval[0] = -deriv[1] / length;
    retval[1] = deriv[0] / length;
    return retval;
}
#endif
