#ifndef __codim1_quadrature_h
#define __codim1_quadrature_h
#include <cmath>
#include "linalg.h"

namespace codim1 {
    typedef std::vector<std::pair<double, double>> QuadratureRule;
    typedef double (*Integrable) (double);

    //TODO: Implement DiligentiMapping for log-singular integrals
    //TODO: Figure out and implement a singularity-killing 

    QuadratureRule double_exp(int n, double h);

    QuadratureRule gauss(unsigned int n);
    
    double integrate(QuadratureRule qr, Integrable fnc);
}
#endif
