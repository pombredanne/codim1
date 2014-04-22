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
#include "integrator.h"

using namespace boost::python;

template<typename T>
void expose_kernel(const char* type_string)
{
    class_<T, bases<Kernel> >(type_string, init<double, double>())
        .def("call", &T::call_all)
        .def("set_interior_data", &T::set_interior_data)
        .def("_call", &T::call)
        .def_readonly("symmetric_matrix", &T::symmetric_matrix)
        .def_readonly("singularity_type", &T::singularity_type);
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

BOOST_PYTHON_MODULE(fast_lib)
{

    //Expose the std::vector interface so it acts like a list/vector/array
    class_<std::vector<double> >("PyVec")
        .def(vector_indexing_suite<std::vector<double> >());
    class_<std::vector<std::vector<double> > >("PyArray")
        .def(vector_indexing_suite<std::vector<std::vector<double> > >());

    // Expose the basis evaluation classes.
    class_<BasisEval, boost::noncopyable>("BasisEval", no_init)
        .def("chain_rule", &BasisEval::chain_rule)
        .def("evaluate", pure_virtual(&BasisEval::evaluate));
    class_<PolyBasisEval, bases<BasisEval> >("PolyBasisEval", 
            init<std::vector<std::vector<double> > >())
        .def("chain_rule", &PolyBasisEval::chain_rule)
        .def("evaluate", &PolyBasisEval::evaluate);
    class_<GradientBasisEval, bases<BasisEval> >("GradientBasisEval", 
            init<std::vector<std::vector<double> > >())
        .def("chain_rule", &GradientBasisEval::chain_rule)
        .def("evaluate", &GradientBasisEval::evaluate);
    class_<FuncEval, bases<BasisEval> >("FuncEval", init<object>())
        .def("chain_rule", &FuncEval::chain_rule)
        .def("evaluate", &FuncEval::evaluate);
    class_<ConstantEval, bases<BasisEval> >("ConstantEval",
            init<std::vector<double> >())
        .def("chain_rule", &ConstantEval::chain_rule)
        .def("evaluate", &ConstantEval::evaluate);
    class_<SolutionEval, bases<BasisEval> >("SolutionEval",
            init<PolyBasisEval&, std::vector<double>,
                 std::vector<std::vector<std::vector<double> > > >())
        .def("chain_rule", &SolutionEval::chain_rule)
        .def("evaluate", &SolutionEval::evaluate);

    // Expose mesh calculation functions
    class_<MeshEval>("MeshEval",
            init<std::vector<std::vector<double> >,
            std::vector<std::vector<double> >,
            std::vector<std::vector<std::vector<double> > > >())
        .def("eval_function", &MeshEval::eval_function)
        .def("get_physical_point", &MeshEval::get_physical_point)
        .def("get_jacobian", &MeshEval::get_jacobian)
        .def("get_normal", &MeshEval::get_normal)
        .def_readonly("basis_eval", &MeshEval::basis_eval)
        .def_readonly("deriv_eval", &MeshEval::deriv_eval);

    // Expose the elastic kernels.
    class_<KernelWrap, boost::noncopyable>("Kernel")
        .def("_call", pure_virtual(&Kernel::call));
    class_<KernelData>("KernelData", no_init)
        .def_readonly("dist", &KernelData::dist);
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
            init<std::vector<double>, std::vector<double> >())
        .def_readonly("x", &QuadratureInfo::x)
        .def_readonly("w", &QuadratureInfo::x);
    def("double_integral", double_integral);
    def("single_integral", single_integral);

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

