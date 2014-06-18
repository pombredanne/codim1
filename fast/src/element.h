#ifndef __codim1_element_h
#define __codim1_element_h

#include <utility>
#include <tuple>
#include <string>
#include "bc.h"

namespace codim1 {
    typedef std::pair<float, float> Vec2;
    typedef std::tuple<Vec2, Vec2> Edge;

    /* Elements consist of endpoints, a mapping for the curve or surface
     * between those endpoints, a basis, and a boundary condition. Elements
     * are the basic building block from which the code solves a boundary
     * value problem.
     */
    class Element {
        public:
            Element() {}
            Element(Edge vertices,
                    unsigned int degree, 
                    BC bc);

            Edge vertices; 
            unsigned int degree;
            std::string bctype;
            BC bc;
            static const int next_dof = 0;
    };
}

#endif
