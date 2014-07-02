#include "quadrature.h"

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
        QuadratureRule retval;
        const double tolerance = 1e-14;
        //Because gaussian quadrature rules are symmetric, I only 
        const unsigned int m = (n+1)/2;
        for (unsigned int i=0;i < m;i++)
        {
            // Initial guess.
            double x = std::cos(M_PI * (i + (3.0/4.0)) / (n + 0.5));

            double dp = 0;
            double dx = 10;
            double p1, p2, p3;

            // Perform newton iterations until the quadrature points
            // have converged.
            while (std::fabs(dx) > tolerance)
            {
                p1 = 1.;
                p2 = 0.;
                for (unsigned int j=0; j<n; ++j)
                {
                    p3 = p2;
                    p2 = p1;
                    p1 = ((2.*j+1.)*x*p2-j*p3)/(j+1);
                }
                dp = (n + 1) * (x*p1-p2)/(x*x-1);
                dx = p1 / dp;
                x = x - dx;
            }

            double w = 2 * (n + 1) * (n + 1) / (n * n * (1 - x * x) * dp * dp);
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
