#include <iostream>
#include <boost/python.hpp>

// class SimpleExample
// {
//    public:
//     SimpleExample() {};
//     void SayHello();
// };
// 
using namespace boost::python;
// Declare our Boost Python Module
// enable_pickling is optional, but I find it useful to
// have persistent storage of simple data types in Python.
BOOST_PYTHON_MODULE(fast_package)
{
    // class_<SimpleExample>("simpleExample",init<>())
    //     .def("SayHello", &SimpleExample::SayHello)
    //     .enable_pickling()
    //     ;
}

// void SimpleExample::SayHello() 
// {
//     std::cout<<"Hello World!";
// }
