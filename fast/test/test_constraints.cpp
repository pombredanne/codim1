#include "catch_with_main.hpp"
#include "constraint.h"
#include "common.h"
#include <armadillo>

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
}

TEST_CASE("Constraint matrix is appended to.", "[constraints]") {
    Constraint c = continuity_constraint(1, 2);
    ConstraintMatrix cm;
    add_constraint(cm, c);
    REQUIRE(cm[1] == c);
}

TEST_CASE("Add with constraints -- simple.", "[constraints]") {
    arma::Mat<double> mymat = arma::zeros<mat>(4,4);
    arma::Col<double> myrhs = arma::zeros<vec>(4);
    ConstraintMatrix cm;
    add_constraint(cm, continuity_constraint(2, 3));
    add_mat_with_constraints(MatrixEntry({1, 1, 2}), mymat, myrhs, cm);
    add_mat_with_constraints(MatrixEntry({2, 1, 2}), mymat, myrhs, cm);
    REQUIRE(AE(mymat(1,1), 2.0));
    REQUIRE(AE(mymat(3,1), 2.0));

    add_mat_with_constraints(MatrixEntry({2, 2, 3.7}), mymat, myrhs, cm);
    REQUIRE(AE(mymat(3,3), 3.7));

    add_rhs_with_constraints(DOFWeight({3, 2.0}), myrhs, cm);
    REQUIRE(AE(myrhs(3), 2.0));

    add_rhs_with_constraints(DOFWeight({2, 2.0}), myrhs, cm);
    REQUIRE(AE(myrhs(3), 4.0));
}

TEST_CASE("Add with constraints -- complex.", "[constraints]") {
    arma::Mat<double> mymat = arma::zeros<mat>(4,4);
    arma::Col<double> myrhs = arma::zeros<vec>(4);

    ConstraintMatrix cm;
    add_constraint(cm, continuity_constraint(2, 3));
}

TEST_CASE("Add with inhomogenous constraints.", "[constraints]") {
    arma::Mat<double> mymat = arma::zeros<mat>(4,4);
    arma::Col<double> myrhs = arma::zeros<vec>(4);

    ConstraintMatrix cm;
    add_constraint(cm, offset_constraint(2, 3, 0.5));

    add_mat_with_constraints(MatrixEntry({1, 2, 2.0}), mymat, myrhs, cm);

    REQUIRE(AE(mymat(1,3), 2.0));
    REQUIRE(AE(myrhs(1), -1.0));
}
