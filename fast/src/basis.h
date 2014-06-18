#ifndef __codim1_basis_eval_h
#define __codim1_basis_eval_h
/* Lagrange polynomial basis over the interval [-1, 1].
 */
class Basis
{
    protected:
        // I intend this class to operate as a Flyweight so I privatize
        // the constructor. It should only be called internally.
        Basis(unsigned int degree);
        std::map<int, Basis> created_objs;
    public:
        int degree;
};
#endif
