#ifndef __codim1_parser_h
#define __codim1_parser_h

#include "yaml-cpp/yaml.h"

namespace codim1 {

    typedef std::pair<float, float> Vec2;
    typedef std::tuple<Vec2, Vec2> Edge;

    class Element {
        public:
            Element() {}
            Element(Edge vertices, unsigned int degree);

            Edge vertices; 
            unsigned int degree;
            static const int next_dof = 0;
    };

    YAML::Node load_file(std::string filename);
    void parse_elements(YAML::Node yaml_elements);
}

namespace YAML {
template<>
struct convert<codim1::Element> {
    static bool decode(const Node& node, codim1::Element& rhs) {
        auto edge = std::make_tuple(node["vertices"][0].as<codim1::Vec2>(),
                                    node["vertices"][1].as<codim1::Vec2>());
        auto degree = node["degree"].as<int>();
        rhs = codim1::Element(edge, degree);
        return true;        
    }
};


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
}


#endif
