#include "UnitTest++.h"
#include "common.h"
#include "quadrature.h"
#include <iostream>

using namespace codim1;

TEST(DoubleExponentialWorks) {
    auto qr = double_exp(8, 0.3); 
    double result = integrate(qr, [](double x) {return std::pow(x, 10) * std::log(x + 1);});
    CHECK_CLOSE(-0.215466, result, 1e-4);

    result = integrate(qr, [](double x) {return std::pow(x, 4) * std::log(x + 1);});
    CHECK_CLOSE(-0.336074, result, 1e-5);

    result = integrate(qr, [](double x) {return std::log(x + 1);});
    CHECK_CLOSE(-0.613706, result, 1e-6);
}

TEST(GaussQuadratureWorks) {
    auto qr = gauss(4);
    CHECK(qr.size() == 4);

    //Check that the weights are in the right range
    for (auto xw: qr) {
        CHECK(std::fabs(xw.first) <= 1);
    }

    double result = integrate(qr, [](double x) {return 3 * x * x;});
    CHECK_CLOSE(2.0, result, 1e-12);

    auto qr_high_order = gauss(50);
    CHECK(qr_high_order.size() == 50);
    result = integrate(qr_high_order, [](double x) {return 101 * std::pow(x, 100);});
    CHECK_CLOSE(2.0, result, 1e-11);

    auto qr_odd = gauss(5);
    CHECK(qr_odd.size() == 5);
}
