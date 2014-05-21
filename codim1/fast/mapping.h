#ifndef __codim1_mesh_eval_h
#define __codim1_mesh_eval_h

#include "basis.h"
#include <vector>

/*
 * Performs fast evaluations for the meshing functions. Returns jacobians
 * physical points, normals.
 */
class MappingEval
{
    public:
        MappingEval(PolyBasis& basis,
                    std::vector<std::vector<double> > coeffs);
        std::vector<double> eval_function(Basis &evaluator, double x_hat);
        std::vector<double> get_physical_point(double x_hat);
        double get_jacobian(double x_hat);
        std::vector<double> get_normal(double x_hat);

        PolyBasis map_basis;
        std::vector<std::vector<double> > coeffs;
};
#endif
