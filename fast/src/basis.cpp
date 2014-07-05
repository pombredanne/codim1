#include "basis.h"
#include <cassert>
#include <stdexcept>
#include <cmath>

namespace codim1 {

    std::vector<double> chebyshev_lobatto_points(unsigned int degree) {
        assert(degree >= 1);
        std::vector<double> x(degree + 1);
        for (unsigned int i = 0; i <= degree; i++) {
            x[i] = std::cos(i * M_PI / degree);
        }
        return x;
    }

    std::vector<double> chebyshev_barycentric_weights(unsigned int degree) {
        assert(degree >= 1);
        std::vector<double> wts(degree + 1);
        wts[0] = 0.5;
        for (unsigned int i = 1; i < degree; i++) {
            wts[i] = (i % 2 == 0) ? 1.0 : -1.0;
        }
        wts[degree] = 0.5;
        return wts;
    }

    Basis::Basis(unsigned int degree):
        degree(degree)
    {
        if (degree < 1) {
            throw std::invalid_argument("The degree must be > 0.");
        }
        nodes = chebyshev_lobatto_points(degree); 
        barycentric_weights = chebyshev_barycentric_weights(degree);
    }

    unsigned int Basis::n_fncs() {
        return degree + 1;
    }

    double Basis::point_eval(std::vector<double> y, double x_hat) {
        double denom = 0;
        double numer = 0;
        for (int i = 0; i <= degree; i++) {
            if (x_hat - nodes[i] < 1e-15) {
                return y[i];
            }
            const double kernel = barycentric_weights[i] / (x_hat - nodes[i]);
            denom += kernel;
            numer += kernel * y[i];
        }
        return numer / denom;
    }
}
