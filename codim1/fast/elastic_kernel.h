#ifndef __codim1_elastic_kernel_h
#define __codim1_elastic_kernel_h
#define PI 3.14159265358979323846
#include <cmath>

struct KernelData
{
    double dist;
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
            reverse_normal = false;
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

        // reverse_normal can be used for considering the opposite side
        // of a crack
        bool reverse_normal;

        // assembly only needs to use the upper triangle (or lower) if
        // the matrix resulting from the kernel is symmetric
        bool symmetric_matrix;

        // we need to use different quadrature formulae for log(r) and 1/r
        // singular kernels.
        std::string singularity_type;
};

KernelData Kernel::get_double_integral_data(std::vector<double> r,
                                    std::vector<double> m, 
                                    std::vector<double> n)
{
    KernelData kd;
    kd.r[0] = r[0];
    kd.r[1] = r[1];
    kd.m[0] = m[0];
    kd.m[1] = m[1];
    kd.n[0] = n[0];
    kd.n[1] = n[1];
    kd.dist = sqrt(pow(r[0], 2) + pow(r[1], 2));
    // grad(r)
    kd.dr[0] = r[0] / kd.dist;
    kd.dr[1] = r[1] / kd.dist;
    // grad(r) dot m
    kd.drdm = kd.dr[0] * m[0] + kd.dr[1] * m[1];
    // grad(r) dot n
    kd.drdn = kd.dr[0] * n[0] + kd.dr[1] * n[1];
    return kd;
}

KernelData Kernel::get_interior_integral_data(std::vector<double> phys_pt,
                                              std::vector<double> m)
{
    std::vector<double> r(2);
    r[0] = soln_point[0] - phys_pt[0];
    r[1] = soln_point[1] - phys_pt[1];
    return get_double_integral_data(r, soln_normal, m);
}

void Kernel::set_interior_data(std::vector<double> soln_point,
                       std::vector<double> soln_normal)
{
    this->soln_point = soln_point;
    this->soln_normal = soln_normal;
}

Kernel::Kernel(double shear_modulus, double poisson_ratio)
{
    this->shear_modulus = shear_modulus;
    this->poisson_ratio = poisson_ratio;
    this->reverse_normal = false;
}

std::vector<std::vector<double> >
Kernel::call_all(std::vector<double> r,
             std::vector<double> m, 
             std::vector<double> n)
{
    KernelData params = get_double_integral_data(r, m, n);
    std::vector<std::vector<double> > retval(2);
    std::vector<double> retval_x(2);
    std::vector<double> retval_y(2);
    retval_x[0] = call(params, 0, 0);
    retval_x[1] = call(params, 0, 1);
    retval_y[0] = call(params, 1, 0);
    retval_y[1] = call(params, 1, 1);

    retval[0] = retval_x;
    retval[1] = retval_y;
    return retval;
}

class MassMatrixKernel: public Kernel
{
    public:
        MassMatrixKernel(double s, double p) {}
        virtual double call(KernelData d, int p, int q);
        virtual KernelData get_interior_integral_data(
                                std::vector<double> r,
                                std::vector<double> m);
};

double MassMatrixKernel::call(KernelData d, int p, int q)
{
    if(p == q)
    {
        return 1.0;
    }
    return 0.0;
}

KernelData MassMatrixKernel::get_interior_integral_data(std::vector<double> r,
                              std::vector<double> m)
{
    KernelData data;
    return data;
}

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

DisplacementKernel::DisplacementKernel(double shear_modulus,
                                   double poisson_ratio)
    :Kernel(shear_modulus, poisson_ratio)
{
    symmetric_matrix = true;
    singularity_type = "logr";
    const3 = 1.0 / (8.0 * PI * shear_modulus * (1 - poisson_ratio));
    const4 = (3.0 - 4.0 * poisson_ratio);
}

double DisplacementKernel::call(KernelData d, int p, int q)
{
    return const3 * (((p == q) * -const4 * log(d.dist)) + 
                     d.dr[p] * d.dr[q]);
}

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

TractionKernel::TractionKernel(double shear_modulus,
                                   double poisson_ratio)
    :Kernel(shear_modulus, poisson_ratio)
{
    symmetric_matrix = false;
    singularity_type = "oneoverr";
    const1 = (1 - 2 * poisson_ratio);
    const2 = 1.0 / (4 * PI * (1 - poisson_ratio));
}

double TractionKernel::call(KernelData d, int p, int q)
{
    return (1 / d.dist) * const2 * (
                -d.drdn * (((p == q) * const1) + 
                           2 * d.dr[p] * d.dr[q]) 
                + ((p != q) * \
                    (const1 * (d.dr[p] * d.n[q] - d.dr[q] * d.n[p]))));
}

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

AdjointTractionKernel::AdjointTractionKernel(double shear_modulus,
                                   double poisson_ratio)
    :Kernel(shear_modulus, poisson_ratio)
{
    symmetric_matrix = false;
    singularity_type = "oneoverr";
    const1 = (1 - 2 * poisson_ratio);
    const2 = 1.0 / (4 * PI * (1 - poisson_ratio));
}

double AdjointTractionKernel::call(KernelData d, int p, int q)
{
    return (1 / d.dist) * const2 * (
                d.drdm * (((p == q) * const1) + 
                           2 * d.dr[p] * d.dr[q]) 
                + ((p != q) * \
                    (const1 * (d.dr[p] * d.m[q] - d.dr[q] * d.m[p]))));
}

/*
*    Gpp -- 1/r^2 singular in 2D, but integration by parts throws two of the
*    (1/r)s onto the basis functions, thus resulting in a log(r) singular
*    kernel.
*    A derivation of this regularization is given in Frangi, Novati, 1996.
*/
class RegularizedHypersingularKernel: public Kernel
{
    public:
        RegularizedHypersingularKernel(double shear_modulus, 
                                       double poisson_ratio);
        virtual double call(KernelData d, int p, int q);

        double const5;
};

RegularizedHypersingularKernel::RegularizedHypersingularKernel(
    double shear_modulus, double poisson_ratio)
    :Kernel(shear_modulus, poisson_ratio)
{
    symmetric_matrix = true;
    singularity_type = "logr";
    const5 = shear_modulus / (2 * PI * (1 - poisson_ratio));
}

double RegularizedHypersingularKernel::call(KernelData d, int p, int q)
{
    return const5 * ((p == q) * log(d.dist) - d.dr[p] * d.dr[q]);
}

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

HypersingularKernel::HypersingularKernel(double shear_modulus,
                                   double poisson_ratio)
    :Kernel(shear_modulus, poisson_ratio)
{
    symmetric_matrix = true;
    singularity_type = "logr";
    const1 = 1 - 2 * poisson_ratio;
    const5 = shear_modulus / (2 * PI * (1 - poisson_ratio));
}

double HypersingularKernel::call(KernelData d, int p, int q)
{
    double i0 = call_inner(d, p, 0, q);
    double i1 = call_inner(d, p, 1, q);

    return i0 * d.m[0] + i1 * d.m[1];
}

double HypersingularKernel::call_inner(KernelData d, int p, int q, int m)
{
    /* UGLY!!! From SGBEM book page 33.*/
    double Spmq = const5 / (d.dist * d.dist);
    Spmq = Spmq * ( 2 * d.drdn * ( const1 * (p==m) * d.dr[q] + \
        poisson_ratio * ( d.dr[p] * (m==q) + d.dr[m] * (p==q) ) - \
        4 * d.dr[p] * d.dr[m] * d.dr[q] ) + \
        2 * poisson_ratio * ( d.n[p] * d.dr[m] * d.dr[q] + \
        d.n[m] * d.dr[p] * d.dr[q] ) + \
        const1 * \
        ( 2 * d.n[q] * d.dr[p] * d.dr[m] + d.n[m] * (p==q) + \
            d.n[p] * (m==q) ) \
        - (1 - 4 * poisson_ratio) * d.n[q] * (p==m));
    return Spmq;
}
#endif
