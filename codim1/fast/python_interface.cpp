/* This C++ package has some of the time critical code for codim1 and
 * is called by the surrounding python package. */ 
#include <iostream>
#include <vector>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include "iterable_converter.h"
#include "basis.h"
#include "mapping.h"
#include "elastic_kernel.h"
#include "integrator.h"
#include "assembly.h"

using namespace boost::python;

template<typename T>
void expose_kernel(const char* type_string)
{
    class_<T, bases<Kernel> >(type_string, init<double, double>())
        .def("call", &T::call_all)
        .def("get_interior_integral_data", &T::get_interior_integral_data)
        .def("_call", &T::call)
        .def_readonly("test_gradient", &Kernel::test_gradient)
        .def_readonly("soln_gradient", &Kernel::soln_gradient)
        .def_readonly("singularity_type", &Kernel::singularity_type)
        .enable_pickling();
}

// In order to subclass Kernel in the python layer, this wrapper struct
// must be created. See:
// http://www.boost.org/doc/libs/1_55_0/libs/python/doc/tutorial/doc/html/python/exposing.html
struct KernelWrap: Kernel, wrapper<Kernel>
{
    virtual double call(KernelData d, int p, int q)
    {
        return this->get_override("_call")(d, p, q);
    }
};

BOOST_PYTHON_MODULE(fast_ext)
{
    // //Expose shared_ptr<Basis>
    register_ptr_to_python<boost::shared_ptr<Basis> >(); 

    //Expose the std::vector interface so it acts like a list/vector/array
    class_<std::vector<double> >("PyVec")
        .def(vector_indexing_suite<std::vector<double> >());
    class_<std::vector<std::vector<double> > >("PyArray")
        .def(vector_indexing_suite<std::vector<std::vector<double> > >());

    // Expose the basis evaluation classes.
    // This could be generalized...
    class_<Basis, boost::noncopyable>("Basis", no_init)
        .def("chain_rule", &Basis::chain_rule)
        .def("evaluate", pure_virtual(&Basis::evaluate_vector))
        .def("get_gradient_basis", &Basis::get_gradient_basis)
        .def_readonly("point_sources", &Basis::point_sources)
        .def_readonly("n_fncs", &Basis::n_fncs);
    class_<PolyBasis, bases<Basis> >("PolyBasis", 
            init<std::vector<std::vector<double> >,
                 std::vector<std::vector<double> >,
                 std::vector<double> >())
        .def("chain_rule", &PolyBasis::chain_rule)
        .def("evaluate", &PolyBasis::evaluate_vector)
        .def_readonly("basis_coeffs", &PolyBasis::basis_coeffs)
        .def_readonly("nodes", &PolyBasis::nodes)
        .enable_pickling();
    class_<CoeffBasis, bases<Basis> >("CoeffBasis", 
            init<PolyBasis&, std::vector<std::vector<double> > >())
        .def("chain_rule", &CoeffBasis::chain_rule)
        .def("evaluate", &CoeffBasis::evaluate_vector)
        .def_readonly("coeffs", &CoeffBasis::coeffs)
        .def_readonly("basis", &CoeffBasis::basis)
        .enable_pickling();
    class_<GradientBasis, bases<PolyBasis> >("GradientBasis", 
            init<std::vector<std::vector<double> >,
                 std::vector<double> >())
        .def("chain_rule", &GradientBasis::chain_rule)
        .def("evaluate", &GradientBasis::evaluate_vector)
        .enable_pickling();
    class_<ConstantBasis, bases<Basis> >("ConstantBasis",
            init<std::vector<double> >())
        .def("chain_rule", &ConstantBasis::chain_rule)
        .def("evaluate", &ConstantBasis::evaluate_vector)
        .enable_pickling();
    class_<ZeroBasis, bases<ConstantBasis> >("ZeroBasis", init<>())
        .enable_pickling();
        // .def("chain_rule", &ZeroBasis::chain_rule)
        // .def("evaluate", &ZeroBasis::evaluate);

    // Expose mesh calculation functions
    class_<MappingEval>("MappingEval",
            init<PolyBasis&, std::vector<std::vector<double> > >())
        .def("eval_function", &MappingEval::eval_function)
        .def("get_physical_point", &MappingEval::get_physical_point)
        .def("get_jacobian", &MappingEval::get_jacobian)
        .def("get_normal", &MappingEval::get_normal)
        .def_readonly("map_basis", &MappingEval::map_basis)
        .enable_pickling();

    // Expose the elastic kernels.
    class_<KernelWrap, boost::noncopyable>("Kernel")
        .def("_call", pure_virtual(&Kernel::call))
        .def("set_interior_data", &Kernel::set_interior_data);
    class_<KernelData>("KernelData", no_init)
        .def_readonly("dist", &KernelData::dist)
        .def_readonly("drdn", &KernelData::drdn)
        .def_readonly("drdm", &KernelData::drdm)
        .def_readonly("dr", &KernelData::dr)
        .def_readonly("n", &KernelData::n)
        .def_readonly("m", &KernelData::m)
        .enable_pickling();
    expose_kernel<MassMatrixKernel>("MassMatrixKernel");
    expose_kernel<DisplacementKernel>("DisplacementKernel");
    expose_kernel<TractionKernel>("TractionKernel");
    expose_kernel<AdjointTractionKernel>("AdjointTractionKernel");
    expose_kernel<HypersingularKernel>("HypersingularKernel");
    expose_kernel<RegularizedHypersingularKernel>
                                ("RegularizedHypersingularKernel");
    expose_kernel<SemiRegularizedHypersingularKernel>
                                ("SemiRegularizedHypersingularKernel");

    //Expose the integration functions.
    class_<QuadratureInfo, boost::noncopyable>("QuadratureInfo", 
            init<double, std::vector<double>, std::vector<double> >())
        .def_readonly("x0", &QuadratureInfo::x0)
        .def_readonly("x", &QuadratureInfo::x)
        .def_readonly("w", &QuadratureInfo::w)
        .enable_pickling();
    def("double_integral", double_integral);
    def("single_integral", single_integral);
    def("aligned_single_integral", aligned_single_integral);

    class_<InteriorPoint, boost::noncopyable>("InteriorPoint", init<>())
        .def("process_element", &InteriorPoint::process_element)
        .def_readonly("result", &InteriorPoint::result);
    class_<AlignedInteriorPoint, bases<InteriorPoint>, boost::noncopyable>
        ("AlignedInteriorPoint", init<>());

    //Misc
    def("basis_speed_test", basis_speed_test);

    // Setup a converter to allow iterable python objects to be converted to
    // std:vector
    iterable_converter()
        .from_python<std::vector<double> >()
        .from_python<std::vector<std::vector<double> > >()
        .from_python<std::vector<std::vector<std::vector<double> > > >()
        .from_python<std::vector<QuadratureInfo> >();
}

