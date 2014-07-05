#include "UnitTest++.h"
#include "common.h"
#include "parser.h"
#include <iostream>

using namespace codim1;
TEST(YamlParserWorks) {
    auto main = load_file("test/data/test_simple.yaml");
    CHECK(main["only_field"].as<std::string>() == "Hello!");
}
    
TEST(LoadingVectorsFromYaml) {
    auto main = load_file("test/data/test_vector.yaml");
    CHECK(main["a_vector"].as<Vec2>() == std::make_pair(3.5f, -4.5f));
}

TEST(LoadingBCFromYaml) {
    auto main = load_file("test/data/test_bc.yaml");
    auto bc = main.as<BC>();
    CHECK(bc.type == "displacement");
    CHECK(bc.degree == 1);
    CHECK(bc.coeffs == PolyCoeffs({0.0, 3.0}));
}

void checkMyElement(Element el, 
                    float v0_0, float v0_1,
                    float v1_0, float v1_1,
                    unsigned int degree, unsigned int bcdegree,
                    PolyCoeffs coeffs,
                    std::string bctype) {
    CHECK(std::get<0>(el.vertices) == std::make_pair(v0_0, v0_1));
    CHECK(std::get<1>(el.vertices) == std::make_pair(v1_0, v1_1));
    CHECK(el.degree == degree);
    CHECK(el.bc.degree == bcdegree);
    CHECK(el.bc.coeffs == coeffs);
    CHECK(el.bc.type == bctype);
}

TEST(LoadingElementFromYaml) {
    auto main = load_file("test/data/test_element.yaml");
    auto el = main.as<Element>();
    checkMyElement(el, 3.0f, 4.0f, 7.0f, 8.0f, 1, 1,
                  PolyCoeffs({-3.0, 1.0}), "displacement");
}

TEST(LoadingElementListFromYaml) {
    auto main = load_file("test/data/test_el_list.yaml");
    auto el_list = main["elements"].as<ElementList>();
    checkMyElement(el_list[0], 3.0f, 4.0f, 7.0f, 8.0f, 1, 1,
                  PolyCoeffs({-3.0, 1.0}), "displacement");
    checkMyElement(el_list[1], 7.0f, 8.0f, 10.0f, 11.1f, 2, 2,
                  PolyCoeffs({-4.0, 1.0}), "traction");
}
