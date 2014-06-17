#ifndef __codim1_element_h
#define __codim1_element_h


class Element
{
    public:
        Element(BC &bc, Mapping &mapping, Basis &basis, DOFList dofs);
        ~Element()
};


#endif
