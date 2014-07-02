#include "catch_with_main.hpp"
#include "common.h"
#include "linalg.h"
#include "basis.h"

using namespace codim1;
TEST_CASE("basis creation works", "[basis]") {
    auto basis_ptr = Basis(1);
    REQUIRE(basis_ptr.get().degree == 1);
    basis_ptr = Basis(5);
    REQUIRE(basis_ptr.get().degree == 5);
}
