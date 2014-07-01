#ifndef __codim1_basis_eval_h
#define __codim1_basis_eval_h

#include <vector>
#include <memory>
#include <boost/flyweight.hpp>
#include <boost/flyweight/key_value.hpp>

/* Lagrange polynomial basis over the interval [-1, 1]. Lagrange polynomials
 * are
 */
namespace codim1 
{

    /* A Lagrange interpolating basis. This class is hidden and not normally
     * used. Use the boost::flyweight that is typedefed to Basis.
     */
    class BasisImpl
    {
        public:
            BasisImpl(unsigned int degree);
            unsigned int nFncs() {return degree + 1;}
            const unsigned int degree;
    };

    typedef boost::flyweights::flyweight<
                boost::flyweights::key_value<unsigned int, BasisImpl>> Basis;
}
#endif
