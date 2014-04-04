#ifndef __codim1_basis_eval_h
#define __codim1_basis_eval_h

#include <vector>
#include <boost/python/object.hpp>
#include <boost/python/extract.hpp>

namespace bp = boost::python;

//Forward declare MeshEval
class MeshEval;

/* Abstract base class (interface) for the basis functions in an
 * integral. 
 */
class BasisEval
{
    public:
        BasisEval() {}

        /* Evaluate a basis function at a point.
         */
        virtual double evaluate(int element_idx,
                                int i, 
                                double x_hat,
                                std::vector<double> x,
                                int d) = 0;
        virtual inline double chain_rule(double jacobian) {return 1.0;}
        virtual std::vector<double> evaluate_vector(int element_idx, int i, 
                                double x_hat,
                                std::vector<double> x) = 0;
};


/* Evaluation class for a single function basis. This allows someone to
 * specify a rhs function or to use a special basis.
 */
class FuncEval: public BasisEval
{
    public:
        FuncEval(bp::object function)
        {
            this->function = function;
        }
        bp::object function;
        virtual double evaluate(int element_idx, int i, 
                                double x_hat,
                                std::vector<double> x,
                                int d);
        virtual std::vector<double> evaluate_vector(int element_idx, int i, 
                                double x_hat,
                                std::vector<double> x);
};

/* Always return a constant value...*/
class ConstantEval: public BasisEval
{
    public:
        ConstantEval(double value)
        {
            this->value = value;
        }

        virtual double evaluate(int element_idx, int i, double x_hat,
                                std::vector<double> x,
                                int d);
        virtual std::vector<double> evaluate_vector(int element_idx, int i, 
                                double x_hat,
                                std::vector<double> x);

        double value;
};

// Basis evaluation could be sped up by manually entering
// the functions for all reasonable degrees.
class PolyBasisEval: public BasisEval
{
    public:
        PolyBasisEval(std::vector<std::vector<double> > basis);
        virtual double evaluate(int element_idx, int i, 
                                double x_hat,
                                std::vector<double> x,
                                int d);
        virtual std::vector<double> evaluate_vector(int element_idx, int i, 
                                double x_hat,
                                std::vector<double> x);

        std::vector<std::vector<double> > basis;
        int order;
};

class SolutionEval: public BasisEval
{
    public:
        SolutionEval(PolyBasisEval &under_basis,
                  std::vector<double> coeffs, 
                  std::vector<std::vector<std::vector<double> > > dof_map);
        virtual double evaluate(int element_idx, int i, 
                                double x_hat,
                                std::vector<double> x,
                                int d);
        virtual std::vector<double> evaluate_vector(int element_idx, int i, 
                                double x_hat,
                                std::vector<double> x);

        PolyBasisEval* under_basis;
        std::vector<double> coeffs;
        std::vector<std::vector<std::vector<double> > > dof_map;
};

/* Includes the spatial arc length derivative... */
class GradientBasisEval: public PolyBasisEval
{
    public:
        GradientBasisEval(std::vector<std::vector<double> > basis):
            PolyBasisEval(basis) {}

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


//A simple test to determine how much overhead the python function calling
//costs
double basis_speed_test(std::vector<std::vector<double> > basis);
#endif
