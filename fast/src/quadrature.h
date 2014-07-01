#ifndef __codim1_quadrature_h
#define __codim1_quadrature_h
#include <cmath>
#include "linalg.h"

namespace codim1 {
    typedef std::vector<std::pair<double, double>> QuadratureRule;
    typedef double (*Integrable) (double);

    //TODO: Implement Gauss
    //TODO: Implement DiligentiMapping for log-singular integrals

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

    // Simplified version of the deal.ii gauss rule. 
    QuadratureRule gauss(unsigned int n) {
        QuadratureRule retval;
        const double tolerance = 1e-14;
        const unsigned int m = (n+1)/2;
        for (unsigned int i=1; i<=m; ++i)
        {
            // Initial guess.
            double x = std::cos(M_PI * (i-.25)/(n+.5));

            double pp;
            double p1, p2, p3;

            // Newton iteration
            do
            {
                // compute L_n(x)
                p1 = 1.;
                p2 = 0.;
                for (unsigned int j=0; j<n; ++j)
                {
                    p3 = p2;
                    p2 = p1;
                    p1 = ((2.*j+1.)*x*p2-j*p3)/(j+1);
                }
                pp = n*(x*p1-p2)/(x*x-1);
                x = x-p1/pp;
            }
            while (abs(p1/pp) > tolerance);

            double w = 2./((1.-x*x)*pp*pp);
            retval.push_back(std::make_pair(x, w));
            retval.push_back(std::make_pair(-x, w));
        }

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
#endif
