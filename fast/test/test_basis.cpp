#include "catch_with_main.hpp"
#include "common.h"
#include "linalg.h"
#include "basis.h"

using namespace codim1;
TEST_CASE("linspace works", "[linspace]") {
    std::vector<double> correct ({0.0, 0.2, 0.4, 0.6, 0.8, 1.0});
    auto test_me = linspace(0.0, 1.0, 6);
    for (std::size_t i = 0; i < test_me.size(); i++) {
        REQUIRE(AE(test_me[i], correct[i]));
    }
}

TEST_CASE("basis creation works", "[basis]") {
    auto basis_ptr = Basis(1);
    REQUIRE(basis_ptr.get().degree == 1);
    basis_ptr = Basis(5);
    REQUIRE(basis_ptr.get().degree == 5);
}
