#include "linalg.h"

namespace codim1 {

    std::vector<double> linspace(double a, double b, int n) {
        std::vector<double> array;
        double step = (b - a) / (n-1);

        while(a <= b) {
            array.push_back(a);
            a += step;           // could recode to better handle rounding errors
        }
        return array;
    }

}
