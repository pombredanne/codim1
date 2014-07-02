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

    std::pair<double, double> legendre_and_n_minus_1(unsigned int n, 
                                                     double x) {
        double p_cur = 1.;
        double p_last = 0.;
        double p_temp;
        for (unsigned int j=0; j<n; ++j)
        {
            p_temp = p_last;
            p_last = p_cur;
            p_cur = ((2.*j+1.)*x*p_last-j*p_temp)/(j+1);
        }
        return std::make_pair(p_cur, p_last);
    }

    QuadratureRule gauss(unsigned int n) {
        QuadratureRule retval(n);
        const double tolerance = 1e-14;
        //Because gaussian quadrature rules are symmetric, I only 
        const unsigned int m = (n+1)/2;
        for (unsigned int i=0;i < m;i++)
        {
            // Initial guess.
            double x = std::cos(M_PI * (i + (3.0/4.0)) / (n + 0.5));

            double dp = 0;
            double dx = 10;

            // Perform newton iterations until the quadrature points
            // have converged.
            while (std::fabs(dx) > tolerance)
            {
                std::pair<double, double> p_n_and_nm1 =
                    legendre_and_n_minus_1(n, x);
                double p_n = p_n_and_nm1.first;
                double p_nm1 = p_n_and_nm1.second;
                dp = (n + 1) * (x * p_n - p_nm1) / (x * x - 1);
                dx = p_n / dp;
                x = x - dx;
            }

            double w = 2 * (n + 1) * (n + 1) / (n * n * (1 - x * x) * dp * dp);
            retval[i] = std::make_pair(x, w);
            retval[n - i - 1] = std::make_pair(-x, w);
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
