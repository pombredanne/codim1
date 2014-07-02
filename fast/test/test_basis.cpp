#include "catch_with_main.hpp"
#include "common.h"
#include "linalg.h"
#include "basis.h"

using namespace codim1;
TEST_CASE("Basis flyweight works", "[basis]") {
    auto basis_ptr = Basis(1);
    REQUIRE(basis_ptr.get().degree == 1);
    basis_ptr = Basis(5);
    REQUIRE(basis_ptr.get().degree == 5);
}

TEST_CASE("Basis nodes correct.", "[basis]") {
    auto basis0 = Basis(0);
    REQUIREAEVECTOR(basis0.get().nodes, std::vector<double>({0.0}), 1e-10);

    auto basis1 = Basis(1); 
    REQUIREAEVECTOR(basis1.get().nodes, std::vector<double>({1.0, -1.0}), 1e-10);

    auto basis2 = Basis(2);
    REQUIREAEVECTOR(basis2.get().nodes, std::vector<double>({1.0, 0.0, -1.0}), 1e-10);

    auto basis10 = Basis(10);
    REQUIREAE(basis10.get().nodes[4], 0.30901699437494745, 1e-10);
}
