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

    std::vector<double> linspace(double a, double b, int n) {
        std::vector<double> array;
        double step = (b-a) / (n-1);

        while(a <= b) {
            array.push_back(a);
            a += step;           // could recode to better handle rounding errors
        }
        return array;
    }
}
#endif
