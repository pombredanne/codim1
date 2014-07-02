#ifndef __codim1_linalg_h
#define __codim1_linalg_h
#include <vector>

namespace arma {
    template <typename T> class Mat;
    template <typename T> class Col;
}

namespace codim1 {
    struct MatrixEntry {
        int row;
        int col;
        double value;
    };
    typedef arma::Mat<double> mat;
    typedef arma::Col<double> vec;
}
#endif
