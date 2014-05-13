#ifndef __codim1_mesh_eval_h
#define __codim1_mesh_eval_h

#include "basis_eval.h"
#include <vector>

/*
 * Performs fast evaluations for the meshing functions. Returns jacobians
 * physical points, normals.
 */
class MappingEval
{
    public:
        MappingEval(std::vector<std::vector<double> > basis,
            std::vector<std::vector<double> > derivs,
            std::vector<std::vector<double> > coeffs);
        std::vector<double> eval_function(BasisEval &evaluator, double x_hat);
        std::vector<double> get_physical_point(double x_hat);
        double get_jacobian(double x_hat);
        std::vector<double> get_normal(double x_hat);

        const std::vector<double> empty_vec;
        PolyBasisEval basis_eval;
        PolyBasisEval deriv_eval;
        std::vector<std::vector<double> > coeffs;
};
#endif
