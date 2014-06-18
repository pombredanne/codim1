#include "element.h"

using namespace codim1;

Element::Element(Edge vertices, unsigned int degree, BC bc) {
    this->vertices = vertices;
    this->degree = degree;
    this->bc = bc;
}

