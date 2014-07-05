#ifndef __codim1_basis_eval_h
#define __codim1_basis_eval_h

#include <vector>

/* Lagrange polynomial basis over the interval [-1, 1]. Lagrange polynomials
 * are the unique polynomials that are 1 at a single interpolation node and
 * zero at all other nodes. 
 * For the nodes, I currently use the Lobatto-Chebyshev nodes.
 * Barycentric lagrange interpolation is used for evaluating the interpolating
 * polynomial. A good introduction to Barycentric Lagrange interpolation is in
 *
 * "Barycentric Lagrange Interpolation"
 * Berrut and Trefethen, 2004, SIAM Review
 *
 */
namespace codim1 
{

    /* A Lagrange interpolating basis.
     */
    class Basis
    {
        public:
            Basis(unsigned int degree);

            unsigned int n_fncs();
            double point_eval(std::vector<double> y, double x_hat);

            const unsigned int degree;
            std::vector<double> nodes;
            std::vector<double> barycentric_weights;
    };
}
#endif
