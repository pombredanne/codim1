#ifndef __codim1_basis_eval_h
#define __codim1_basis_eval_h
#include <vector>
#include <boost/python/object.hpp>
#include <boost/python/extract.hpp>


namespace bp = boost::python;

/* Abstract base class (interface) for the basis functions in an
 * integral. 
 */
class BasisEval
{
    public:
        BasisEval() {}

        /* Evaluate a basis function at a point.
         */
        virtual double evaluate(int i, 
                                double x_hat,
                                std::vector<double> x,
                                int d) = 0;
        virtual std::vector<double> evaluate_vector(int i, 
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
        virtual double evaluate(int i, 
                                double x_hat,
                                std::vector<double> x,
                                int d);
        virtual std::vector<double> evaluate_vector(int i, 
                                double x_hat,
                                std::vector<double> x);
};

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

// Basis evaluation could be sped up by manually entering
// the functions for all reasonable degrees.
class PolyBasisEval: public BasisEval
{
    public:
        PolyBasisEval(std::vector<std::vector<double> > basis);
        virtual double evaluate(int i, 
                                double x_hat,
                                std::vector<double> x,
                                int d);
        virtual std::vector<double> evaluate_vector(int i, 
                                double x_hat,
                                std::vector<double> x);

        std::vector<std::vector<double> > basis;
        int order;
};

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

//A simple test to determine how much overhead the python function calling
//costs
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
#endif
