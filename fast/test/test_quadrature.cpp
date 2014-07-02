#include "catch_with_main.hpp"
#include "common.h"
#include "quadrature.h"

using namespace codim1;

TEST_CASE("Double exponential works.", "[quadrature]") {
    auto qr = double_exp(8, 0.3); 
    double result = integrate(qr, [](double x) {return std::pow(x, 10) * std::log(x + 1);});
    REQUIREAE(-0.215466, result, 1e-4);

    result = integrate(qr, [](double x) {return std::pow(x, 4) * std::log(x + 1);});
    REQUIREAE(-0.336074, result, 1e-5);

    result = integrate(qr, [](double x) {return std::log(x + 1);});
    REQUIREAE(-0.613706, result, 1e-6);
}

TEST_CASE("Gauss quadrature works.", "[quadrature]") {
    auto qr = gauss(4);

    //Check that the weights are in the right range
    for (auto xw: qr) {
        REQUIRE(std::fabs(xw.first) <= 1);
        std::cout << "x: " << xw.first << "   w: " << xw.second << std::endl;
    }

    double result = integrate(qr, [](double x) {return 3 * x * x;});
    REQUIREAE(2.0, result, 1e-12);

    auto qr_high_order = gauss(50);
    result = integrate(qr_high_order, [](double x) {return 101 * std::pow(x, 100);});
    REQUIREAE(2.0, result, 1e-11);
}
