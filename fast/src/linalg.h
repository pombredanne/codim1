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
