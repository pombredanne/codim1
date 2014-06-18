#include "bc.h"

using namespace codim1;

BC::BC(std::string bctype, int basis_degree, PolyCoeffs coeffs) {
    this->type = bctype;
    this->degree = basis_degree;
    this->coeffs = coeffs;
}
