#ifndef __codim1_elastic_kernel_h
#define __codim1_elastic_kernel_h
#define PI 3.14159265358979323846
#include <cmath>
#include <vector>
#include <string>

struct KernelData
{
    double dist;
    double dist2;
    double drdn;
    double drdm;
    double r[2];
    double dr[2];
    double n[2];
    double m[2];
};

class Kernel
{
    public:
        Kernel() 
        {
            shear_modulus = 1.0;
            poisson_ratio = 0.25;
        }

        // All a kernel needs to know are the elastic parameters.
        Kernel(double shear_modulus, double poisson_ratio);

        //Calculate the parameters and return the full kernel.
        virtual std::vector<std::vector<double> > call_all(
            std::vector<double> r, 
            std::vector<double> m, 
            std::vector<double> n);

        //Returns only one of the kernel elements.
        virtual double call(KernelData d, int p, int q) = 0;

        virtual KernelData get_double_integral_data(std::vector<double> r,
                                            std::vector<double> m, 
                                            std::vector<double> n);

        virtual KernelData get_interior_integral_data(std::vector<double> r,
                                              std::vector<double> m);
        
        void set_interior_data(std::vector<double> soln_point,
                               std::vector<double> soln_normal);
        // These are used when computing interior integrals.
        std::vector<double> soln_point;
        std::vector<double> soln_normal;

        double shear_modulus;
        double poisson_ratio;

        // assembly only needs to use the upper triangle (or lower) if
        // the matrix resulting from the kernel is symmetric
        bool symmetric_matrix;

        // we need to use different quadrature formulae for log(r) and 1/r
        // singular kernels.
        std::string singularity_type;
};

class MassMatrixKernel: public Kernel
{
    public:
        MassMatrixKernel(double s, double p) {}
        virtual double call(KernelData d, int p, int q);
        virtual KernelData get_interior_integral_data(
                                std::vector<double> r,
                                std::vector<double> m);
};

/*
*    Guu -- log(r) singular in 2D
*/
class DisplacementKernel: public Kernel
{
    public:
        DisplacementKernel(double shear_modulus, double poisson_ratio);
        virtual double call(KernelData d, int p, int q);

        double const3;
        double const4;
};


/*
* Gup -- 1/r singular in 2D
*/
class TractionKernel: public Kernel
{
    public:
        TractionKernel(double shear_modulus, double poisson_ratio);
        virtual double call(KernelData d, int p, int q);

        double const1;
        double const2;
};


/*
*   Gpu -- 1/r singular in 2D
*   Exactly the same code as the standard TractionKernel, just with the
*   a different sign and the relevant normal being m instead of n.
*/
class AdjointTractionKernel: public Kernel
{
    public:
        AdjointTractionKernel(double shear_modulus, double poisson_ratio);
        virtual double call(KernelData d, int p, int q);

        double const1;
        double const2;
};

/*
*    Gpp -- 1/r^2 singular in 2D, but integration by parts throws two of the
*    (1/r)s onto the basis functions, thus resulting in a log(r) singular
*    kernel.
*    A derivation of this regularization is given in Frangi, Novati, 1996.
*    This kernel cannot be used for interior point computation, because it
*    is only properly defined when within a double integral.
*/
class RegularizedHypersingularKernel: public Kernel
{
    public:
        RegularizedHypersingularKernel(double shear_modulus, 
                                       double poisson_ratio);
        virtual double call(KernelData d, int p, int q);

        double const5;
};

/*
* A half regularized form of the hypersingular kernel. The kernel
* is strongly singular and is based on the normal derivatives of the
* regularized traction kernel
* This kernel can be used for interior point computation when the 
* interior point is near to the boundary. The singularity is less 
* intense than for the original hypersingular equation. 
* This kernel can also be used when a point source displacement discontinuity
* is applied, like a step function displacement discontinuity boundary
* condition.
*/
class SemiRegularizedHypersingularKernel: public Kernel
{
    public:
        SemiRegularizedHypersingularKernel(double shear_modulus, 
                                       double poisson_ratio);
        virtual double call(KernelData d, int p, int q);

        double const5;
        const double e[2][2] = {{0, 1},{-1, 0}};
};

/*
*   The non-regularized hypersingular kernel. Useful for interior 
*   computation, though it will be problematic when the interior point
*   is very close to the boundary.
*/
class HypersingularKernel: public Kernel
{
    public:
        HypersingularKernel(double shear_modulus, double poisson_ratio);
        virtual double call(KernelData d, int p, int q);
        double call_inner(KernelData d, int p, int q, int m);

        double const1;
        double const5;
};
#endif
