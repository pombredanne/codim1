/* This C++ package has some of the time critical code for codim1 and
 * is called by the surrounding python package. */ 
#include <iostream>
#include <vector>
#include <list>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <math.h>

namespace bp = boost::python;

/// I took this from http://stackoverflow.com/questions/15842126/feeding-a-python-list-into-a-function-taking-in-a-vector-with-boost-python
/// @brief Type that allows for registration of conversions from
///        python iterable types.
struct iterable_converter
{
/// @note Registers converter from a python interable type to the
///       provided type.
    template <typename Container> iterable_converter& from_python()
    {
    bp::converter::registry::push_back(
    &iterable_converter::convertible,
    &iterable_converter::construct<Container>,
    bp::type_id<Container>());
    return *this;
    }

    /// @brief Check if PyObject is iterable.
    static void* convertible(PyObject* object)
    {
        return PyObject_GetIter(object) ? object : NULL;
    }

    /// @brief Convert iterable PyObject to C++ container type.
    ///
    /// Container Concept requirements:
    ///
    ///   * Container::value_type is CopyConstructable.
    ///   * Container can be constructed and populated with two iterators.
    ///     I.e. Container(begin, end)
    template <typename Container>
    static void construct(
    PyObject* object,
    bp::converter::rvalue_from_python_stage1_data* data)
    {
        namespace python = bp;
        // Object is a borrowed reference, so create a handle indicting 
        // it is
        // borrowed for proper reference counting.
        bp::handle<> handle(bp::borrowed(object));

        // Obtain a handle to the memory block that the converter 
        // has allocated
        // for the C++ type.
        typedef bp::converter::rvalue_from_python_storage<Container>
                                                     storage_type;
        void* storage = 
            reinterpret_cast<storage_type*>(data)->storage.bytes;

        typedef bp::stl_input_iterator<typename Container::value_type>
                                                         iterator;

        // Allocate the C++ type into the converter's memory block, and 
        // assign
        // its handle to the converter's convertible variable.  The C++
        // container is populated by passing the begin and end iterators of
        // the python object to the container's constructor.
        data->convertible = new (storage) Container(
            iterator(bp::object(handle)), // begin
            iterator());                      // end
    }
};


// Basis evaluation could be sped up by manually entering
// the functions for all reasonable degrees.
// Also, array indexing would be more efficient if the
// pyublas interface were moved out of the way, use python lists
// at initialization and then store in a std::vector
class BasisEval
{
    public:
        BasisEval(std::vector<std::vector<double> > basis);

        /* Evaluate a basis function at a point.
         */
        double evaluate(int i, double x);

        std::vector<std::vector<double> > basis;
        int order;
};

BasisEval::BasisEval(std::vector<std::vector<double> > basis)
{
    this->basis = basis;
    this->order = basis.size();
}

double BasisEval::evaluate(int i, double x)
{
    double running_mult = 1.0;
    double retval = basis[i][order - 1];
    for(int coeff_idx = order - 2; coeff_idx >= 0; coeff_idx--)
    {
        running_mult *= x;
        retval += basis[i][coeff_idx] * running_mult;
    }
    return retval;
}

void basis_speed_test(std::vector<std::vector<double> > basis)
{
    BasisEval b(basis);
    for(int i = 0; i < 1000000; i++)
    {
        double a = b.evaluate(1, 0.5);
    }
}

using namespace boost::python;

BOOST_PYTHON_MODULE(fast_package)
{
    // Setup a converter to allow iterable python objects to be converted to
    // std:vector
    iterable_converter()
        .from_python<std::vector<double> >()
        .from_python<std::vector<std::vector<double> > >();

    class_<BasisEval>("BasisEval", 
            init<std::vector<std::vector<double> > >())
        .def("evaluate", &BasisEval::evaluate);
    def("basis_speed_test", basis_speed_test);
}

