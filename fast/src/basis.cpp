#include "basis.h"
#include "quadrature.h"
#include <cmath>

namespace codim1 {

    std::vector<double> chebyshev_lobatto_points(unsigned int degree) {
        std::vector<double> x(degree + 1);
        for (unsigned int i = 0; i <= degree; i++) {
            x[i] = std::cos(i * M_PI / degree);
        }
        return x;
    }

    BasisImpl::BasisImpl(unsigned int degree):
        degree(degree)
    {
        
        if (degree == 0) {
            nodes = {0.0};
        } else {
            nodes = chebyshev_lobatto_points(degree); 
        }
    }

}
