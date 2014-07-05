#include "UnitTest++.h"
#include "basis.h"
#include <stdexcept>

using namespace codim1;
TEST(BasisConstruction) {
    auto basis_ptr = Basis(1);
    CHECK(basis_ptr.degree == 1);
    auto basis_ptr2 = Basis(5);
    CHECK(basis_ptr2.degree == 5);
}

TEST(BasisFncsCorrect) {
    CHECK(Basis(2).n_fncs() == 3);
}

TEST(NoZeroDegreeBasis) {
    CHECK_THROW(Basis(0), std::invalid_argument);
}

TEST(BasisNodesCorrect) {
    auto basis1 = Basis(1); 
    CHECK_ARRAY_CLOSE(basis1.nodes,
                      std::vector<double>({1.0, -1.0}),
                      2,
                      1e-14);

    auto basis2 = Basis(2);
    CHECK_ARRAY_CLOSE(basis2.nodes, 
                      std::vector<double>({1.0, 0.0, -1.0}),
                      3,
                      1e-14);

    auto basis10 = Basis(10);
    CHECK_CLOSE(basis10.nodes[4], 0.30901699437494745, 1e-14);
}

TEST(BasisBarycentricWeightsCorrect) {
    auto basis5 = Basis(5);

    CHECK(basis5.barycentric_weights.size() == 6);
    CHECK_ARRAY_CLOSE(std::vector<double>({0.5, -1, 1, -1, 1, 0.5}),
                      basis5.barycentric_weights, 6, 1e-14);
}

TEST(BasisEvalConstant) {
    auto basis2 = Basis(2);
    double val = basis2.point_eval(std::vector<double>({1.0, 1.0, 1.0}), 0.0);
    CHECK_CLOSE(val, 1.0, 1e-15);
}

TEST(BasisEvalSingleNode) {
    auto basis2 = Basis(2);
    double val = basis2.point_eval(std::vector<double>({1.0, 0.0, 0.0}), 
                                   basis2.nodes[1]);
    CHECK_CLOSE(val, 0.0, 1e-15);
}
