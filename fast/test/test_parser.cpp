#include "common.h"
#include "parser.h"
#include "catch_with_main.hpp"
#include <iostream>

using namespace codim1;
TEST_CASE("yaml parser works", "[yaml]") {
    auto main = load_file("test/data/test_simple.yaml");
    REQUIRE(main["only_field"].as<std::string>() == "Hello!");
}
    
TEST_CASE("loading vectors from yaml", "[yaml]") {
    auto main = load_file("test/data/test_vector.yaml");
    REQUIRE(main["a_vector"].as<Vec2>() == std::make_pair(3.5f, -4.5f));
}

TEST_CASE("loading bc from yaml", "[yaml]") {
    auto main = load_file("test/data/test_bc.yaml");
    auto bc = main.as<BC>();
    REQUIRE(bc.type == "displacement");
    REQUIRE(bc.degree == 1);
    REQUIRE(bc.coeffs == PolyCoeffs({0.0, 3.0}));
}

void checkMyElement(Element el, 
                    float v0_0, float v0_1,
                    float v1_0, float v1_1,
                    int degree, int bcdegree,
                    PolyCoeffs coeffs,
                    std::string bctype) {
    REQUIRE(std::get<0>(el.vertices) == std::make_pair(v0_0, v0_1));
    REQUIRE(std::get<1>(el.vertices) == std::make_pair(v1_0, v1_1));
    REQUIRE(el.degree == degree);
    REQUIRE(el.bc.degree == bcdegree);
    REQUIRE(el.bc.coeffs == coeffs);
    REQUIRE(el.bc.type == bctype);
}

TEST_CASE("loading element from yaml", "[yaml]") {
    auto main = load_file("test/data/test_element.yaml");
    auto el = main.as<Element>();
    checkMyElement(el, 3.0f, 4.0f, 7.0f, 8.0f, 1, 1,
                  PolyCoeffs({-3.0, 1.0}), "displacement");
}

TEST_CASE("loading element list from yaml", "[yaml]") {
    auto main = load_file("test/data/test_el_list.yaml");
    auto el_list = main["elements"].as<ElementList>();
    checkMyElement(el_list[0], 3.0f, 4.0f, 7.0f, 8.0f, 1, 1,
                  PolyCoeffs({-3.0, 1.0}), "displacement");
    checkMyElement(el_list[1], 7.0f, 8.0f, 10.0f, 11.1f, 2, 2,
                  PolyCoeffs({-4.0, 1.0}), "traction");
}
