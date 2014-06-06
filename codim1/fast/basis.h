#ifndef __codim1_basis_eval_h
#define __codim1_basis_eval_h

#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/python/object.hpp>
#include <boost/python/extract.hpp>

namespace bp = boost::python;

/*
This stuff should be reworked so that basis functions exist elementwise.

Use another function like apply_mapping that applies a basis function to each member of an iterable.

This is partially (mostly) done.
*/

static const std::vector<double> empty_vec(2, 0.0);
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
                                std::vector<double> x,
                                int d) = 0;

        /* Simple evaluator for a case that doesn't depend on the physical
         * point. Do not use this with the SingleFunctionBasis.
         * Passes zeros for the physical point.
         */
        double evaluate_simple(int i, double x_hat, int d)
        {return evaluate_internal(i, x_hat, empty_vec, d);}

        virtual inline double chain_rule(double jacobian) {return 1.0;}
        virtual std::vector<double> evaluate_vector(int i, 
                                double x_hat,
                                std::vector<double> x) = 0;

        boost::shared_ptr<Basis> get_gradient_basis()
        {
            return gradient_basis;
        }

        boost::shared_ptr<Basis> get_point_sources()
        {
            
        }

        boost::shared_ptr<Basis> gradient_basis;
        int n_fncs;
};


/* Evaluation class for a single function basis. This allows someone to
 * specify a rhs function or to use a special basis.
 * TODO: Get rid of this. Only allow polynomial bases.
 */
class SingleFunctionBasis: public Basis
{
    public:
        SingleFunctionBasis(bp::object function)
        {
            this->n_fncs = 1;
            this->function = function;
        }
        bp::object function;
        virtual double evaluate_internal(int i, 
                                double x_hat,
                                std::vector<double> x,
                                int d);
        virtual std::vector<double> evaluate_vector(int i, 
                                double x_hat,
                                std::vector<double> x);
};

/* Always return a constant value...*/
class ConstantBasis: public Basis
{
    public:
        ConstantBasis(std::vector<double> values)
        {
            this->n_fncs = 1;
            this->values = values;
        }

        virtual double evaluate_internal(int i, double x_hat,
                                std::vector<double> x,
                                int d);
        virtual std::vector<double> evaluate_vector(int i, 
                                double x_hat,
                                std::vector<double> x);

        std::vector<double> values;
};

/* Always return zero. Distinguished from ConstantBasis so that
 * function can check type and not perform any computation at all in the 
 * special case of zero.  */
class ZeroBasis: public ConstantBasis
{
    public:
        ZeroBasis():
            ConstantBasis(std::vector<double>(2, 0.0))
        { }
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
                  std::vector<double> nodes);
        virtual double evaluate_internal(int i, 
                                double x_hat,
                                std::vector<double> x,
                                int d);
        virtual std::vector<double> evaluate_vector(int i, 
                                double x_hat,
                                std::vector<double> x);

        std::vector<std::vector<double> > basis_coeffs;
        std::vector<double> nodes;
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
class SolutionBasis: public Basis
{
    public:
        SolutionBasis(PolyBasis& basis,
                      std::vector<std::vector<double> > coeffs);

        virtual double evaluate_internal(int i, 
                                double x_hat,
                                std::vector<double> x,
                                int d);
        std::vector<double> evaluate_vector(int i, double x_hat,
                                            std::vector<double> x);

        PolyBasis* basis;
        // 2d array. First axis is x or y. Second axis 
        std::vector<std::vector<double> > coeffs;
};



//A simple test to determine how much overhead the python function calling
//costs
double basis_speed_test(std::vector<std::vector<double> > basis);
#endif
