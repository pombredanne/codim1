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

        const std::vector<double> empty_vec;
        PolyBasisEval basis_eval;
        PolyBasisEval deriv_eval;
        std::vector<std::vector<std::vector<double> > > coeffs;
};
#endif
