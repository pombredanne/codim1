#ifndef __codim1_parser_h
#define __codim1_parser_h

#include "yaml-cpp/yaml.h"
#include "element.h"
#include "bc.h"


namespace codim1 {
    YAML::Node load_file(std::string filename);
    void parse_elements(YAML::Node yaml_elements);
    typedef std::vector<Element> ElementList;
}

namespace YAML {
    template<>
    struct convert<codim1::Vec2> {
        static bool decode(const Node& node, codim1::Vec2& rhs) {
            if(!node.IsSequence() || node.size() != 2) {
                return false;
            }

            rhs = std::make_pair(node[0].as<float>(), 
                                 node[1].as<float>());
            return true;
        }
    };

    template<>
    struct convert<codim1::BC> {
        static bool decode(const Node& node, codim1::BC& rhs) {
            auto type = node["type"].as<std::string>();
            auto degree = node["degree"].as<int>();
            auto coeffs = node["coeffs"].as<codim1::PolyCoeffs>();
            rhs = codim1::BC(type, degree, coeffs);
            return true;        
        }
    };

    template<>
    struct convert<codim1::Element> {
        static bool decode(const Node& node, codim1::Element& rhs) {
            auto edge = std::make_tuple(node["vertices"][0].as<codim1::Vec2>(),
                                        node["vertices"][1].as<codim1::Vec2>());
            auto degree = node["degree"].as<int>();
            auto bc = node["bc"].as<codim1::BC>();
            rhs = codim1::Element(edge, degree, bc);
            return true;        
        }
    };
}
#endif
