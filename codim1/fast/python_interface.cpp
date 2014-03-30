/* This C++ package has some of the time critical code for codim1 and
 * is called by the surrounding python package. */ 
#include <iostream>
#include <vector>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include "iterable_converter.h"
#include "basis_eval.h"
#include "mesh_eval.h"
#include "elastic_kernel.h"

using namespace boost::python;

template<typename T>
void expose_kernel(const char* type_string)
{
    class_<T>(type_string, init<double, double>())
        .def("call", &T::call_all)
        .def("_call", &T::call)
        .def_readonly("reverse_normal", &T::reverse_normal)
        .def_readonly("symmetric_matrix", &T::symmetric_matrix)
        .def_readonly("singularity_type", &T::singularity_type);
}

BOOST_PYTHON_MODULE(fast_lib)
{
    // Setup a converter to allow iterable python objects to be converted to
    // std:vector
    iterable_converter()
        .from_python<std::vector<double> >()
        .from_python<std::vector<std::vector<double> > >()
        .from_python<std::vector<std::vector<std::vector<double> > > >();

    //Expose the std::vector interface so it acts like a list
    boost::python::class_<std::vector<double> >("PyVec")
            .def(boost::python::vector_indexing_suite<
                    std::vector<double> >());

    boost::python::class_<std::vector<std::vector<double> > >("PyArray")
            .def(boost::python::vector_indexing_suite<
                    std::vector<std::vector<double> > >());

    //Expose the evaluation classes.
    class_<BasisEval>("BasisEval", 
            init<std::vector<std::vector<double> > >())
        .def("evaluate", &BasisEval::evaluate);

    class_<MeshEval>("MeshEval",
            init<std::vector<std::vector<double> >,
            std::vector<std::vector<double> >,
            std::vector<std::vector<std::vector<double> > > >())
        .def("get_physical_point", &MeshEval::get_physical_point)
        .def("get_jacobian", &MeshEval::get_jacobian)
        .def("get_normal", &MeshEval::get_normal);

    expose_kernel<DisplacementKernel>("DisplacementKernel");
    expose_kernel<TractionKernel>("TractionKernel");
    expose_kernel<AdjointTractionKernel>("AdjointTractionKernel");
    expose_kernel<HypersingularKernel>("HypersingularKernel");
    expose_kernel<RegularizedHypersingularKernel>
                                ("RegularizedHypersingularKernel");

    def("basis_speed_test", basis_speed_test);
}

