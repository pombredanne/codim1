#include "quadrature.h"
#include <gsl/gsl_integration.h>

namespace codim1 {
    QuadratureRule double_exp(int n, double h) {
        QuadratureRule retval;
        for (int i = -n; i <= n; i++) {
            const double sinhterm = 0.5 * M_PI * std::sinh(i * h);
            const double coshsinh = std::cosh(sinhterm);
            const double x = std::tanh(sinhterm);
            const double w = h * 0.5 * M_PI * std::cosh(i * h) / (coshsinh * coshsinh);
            retval.push_back(std::make_pair(x, w));
        }
        return retval;
    }

    QuadratureRule gauss(unsigned int n) {
        gsl_integration_glfixed_table* table = \
            gsl_integration_glfixed_table_alloc(n);
        double x;
        double w;
        QuadratureRule retval(n);
        for (unsigned int i = 0; i < n; i++) {
            gsl_integration_glfixed_point(-1, 1, i, &x, &w, table);
            retval[i] = std::make_pair(x, w);
        }
        gsl_integration_glfixed_table_free(table);
        return retval;
    }

    double integrate(QuadratureRule qr, Integrable fnc) {
        double integral_val = 0;
        for (auto xw: qr) {
            integral_val += xw.second * fnc(xw.first);
        }
        return integral_val;
    }
}
