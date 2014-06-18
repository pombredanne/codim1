#ifndef __codim1_basis_eval_h
#define __codim1_basis_eval_h

#include <vector>
#include <memory>

/*
Bases exists elementwise.
*/


/* Abstract base class (interface) for the basis functions in an
 * integral. 
 */
class Basis
{
    public:
        Basis() {}

        /* Evaluate a basis function at a point.
         */
        virtual double evaluate_internal(int i, 
                                double x_hat,
                                int d) = 0;

        /* Simple evaluator for a case that doesn't depend on the physical
         * point. Do not use this with the SingleFunctionBasis.
         * Passes zeros for the physical point.
         */
        double evaluate_simple(int i, double x_hat, int d)
        {return evaluate_internal(i, x_hat, d);}

        virtual inline double chain_rule(double jacobian) {return 1.0;}
        virtual std::vector<double> evaluate_vector(int i, double x_hat) = 0;

        std::shared_ptr<Basis> get_gradient_basis()
        {
            return gradient_basis;
        }

        std::vector<std::vector<double> > point_sources;
        std::vector<int> point_source_dependency;
        std::shared_ptr<Basis> gradient_basis;
        int n_fncs;
};

/*
*   This class handles interactions with Lagrange polynomials defined on
*   the unit reference interval [0, 1].
*   The basis functions are defined such that
*   \hat{\phi}_i(\hat{x}) = \phi_i(x)
*   In other words, there is no transformation factor between reference
*   and physical space. So, there is no chain rule contribution.
*/
// Basis evaluation could be sped up by precomputing the values.
class PolyBasis: public Basis
{
    public:
        PolyBasis(std::vector<std::vector<double> > basis_coeffs,
                  std::vector<double> nodes);
        PolyBasis(std::vector<std::vector<double> > basis_coeffs,
                  std::vector<std::vector<double> > basis_derivs,
                  std::vector<std::vector<double> > point_sources,
                  std::vector<int> point_source_dependency,
                  std::vector<double> nodes);
        virtual double evaluate_internal(int i, 
                                double x_hat,
                                int d);
        virtual std::vector<double> evaluate_vector(int i, 
                                double x_hat);

        std::vector<std::vector<double> > basis_coeffs;
        std::vector<double> nodes;

        static const PolyBasis one = PolyBasis({{1.0}, {1.0}}, {0.5});
};

/*
*   Stores the derivatives of the basis functions. For internal use only.
*   Because the derivative is now defined on the reference triangle, the
*   transformation from physical to reference space requires an application
*   of the chain rule. This gives an extra term d\hat{x}/dx. Thus, these
*   basis functions cannot be defined except in reference to a specific
*   mesh.
*/
/* Includes the spatial arc length derivative... */
class GradientBasis: public PolyBasis
{
    public:
        GradientBasis(std::vector<std::vector<double> > basis,
                      std::vector<double> nodes):
            PolyBasis(basis, nodes) {}

        /* 
        * Returns the basis derivative including the
        * scaling to convert a arc length derivative to a basis
        * function derivative.
        */ 
        virtual inline double chain_rule(double jacobian)
        {
            return 1.0 / jacobian;
        }
};

/* A basis that stores solution coefficients alongside the basis functions
 * so that post-processing can be performed on the polynomial solution.
 */
class CoeffBasis: public Basis
{
    public:
        CoeffBasis(Basis& basis,
                      std::vector<std::vector<double> > coeffs);

        virtual double evaluate_internal(int i, 
                                double x_hat,
                                int d);
        std::vector<double> evaluate_vector(int i, double x_hat);
        virtual inline double chain_rule(double jacobian)
        {
            return basis->chain_rule(jacobian);
        }

        Basis* basis;
        // 2d array. First axis is x or y. Second axis is the dof idx
        std::vector<std::vector<double> > coeffs;
};



//A simple test to determine how much overhead the python function calling
//costs
double basis_speed_test(std::vector<std::vector<double> > basis);
#endif
