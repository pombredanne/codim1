#ifndef __codim1_bc_h
#define __codim1_bc_h
#include <vector>
#include <string>

namespace codim1 {
    /* Poly coeffs should be ordered from low degree to
     * high degree. In other words, if 
     * auto a = {1.0, 2.0, 3.0}, the polynomial represented is 
     * 1 + 2x + 3x^2
     */
    typedef std::vector<float> PolyCoeffs;

    //TODO: Change the boundary condition string into an enum or something
    //like that. Whatever is appropriate for C++ good practices. Google this.
    /* Boundary conditions consist of a type -- which is a string, and
     * a basis with coefficients.
     */
    class BC {
        public:
            BC() {}
            BC(std::string bctype, int basis_degree, PolyCoeffs coeffs);

            std::string type;
            /* The degree + polynomial coefficients pairing is a basis function
             * set. I should use the appropriate abstractions. And, modify the
             * basis functions as necessary.
             */
            unsigned int degree;
            PolyCoeffs coeffs;
    };
}
#endif
