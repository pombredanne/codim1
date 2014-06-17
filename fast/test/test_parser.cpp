#include "catch_with_main.hpp"

#include "parser.h"

using namespace codim1;
TEST_CASE("yaml mesh-spec is parsed", "[yaml]") {
    auto main = load_file("test/data/test_simple.yaml");
    REQUIRE(main["only_field"].as<std::string>() == "Hello!");
    
    main = load_file("test/data/test_vector.yaml");
    REQUIRE(main["a_vector"].as<Vec2>() == std::make_pair(3.5f, -4.5f));

    main = load_file("test/data/test_element.yaml");
    auto el = main.as<Element>();
    REQUIRE(std::get<0>(el.vertices) == std::make_pair(3.0f, 4.0f));
    REQUIRE(std::get<1>(el.vertices) == std::make_pair(7.0f, 8.0f));
    REQUIRE(el.degree == 1);
}
