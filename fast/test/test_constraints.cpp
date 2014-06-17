#include "catch_with_main.hpp"

#include "constraint.h"
using namespace codim1;

TEST_CASE("Constraints are created", "[constraints]") {
    Constraint c = continuity_constraint(1, 2);
    Constraint correct = {
        DOFWeight(1, 1.0f),
        DOFWeight(2, -1.0f)
    };
    REQUIRE(c == correct);

    Constraint c2 = offset_constraint(3, 4, 5.0f);
    correct = {
        DOFWeight(3, 1.0f),
        DOFWeight(4, -1.0f),
        DOFWeight(RHS, 5.0f)
    };
    REQUIRE(c2 == correct);

    ConstraintList c_list = {c, c2};
    REQUIRE(c_list[0] == c);
    REQUIRE(c_list[1] == c2);
}
