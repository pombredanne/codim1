#include "elastic_kernel.h"

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
    kd.dist2 = pow(r[0], 2) + pow(r[1], 2);
    kd.dist = sqrt(kd.dist2);
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
                                              std::vector<double> n)
{
    std::vector<double> r(2);
    r[0] = soln_point[0] - phys_pt[0];
    r[1] = soln_point[1] - phys_pt[1];
    return get_double_integral_data(r, soln_normal, n);
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

DisplacementKernel::DisplacementKernel(double shear_modulus,
                                   double poisson_ratio)
    :Kernel(shear_modulus, poisson_ratio)
{
    singularity_type = "logr";
    test_gradient = false;
    soln_gradient = false;
    const3 = 1.0 / (8.0 * PI * shear_modulus * (1 - poisson_ratio));
    const4 = (3.0 - 4.0 * poisson_ratio);
}

double DisplacementKernel::call(KernelData d, int p, int q)
{
    return const3 * (((p == q) * -const4 * log(d.dist)) + 
                     d.dr[p] * d.dr[q]);
}

TractionKernel::TractionKernel(double shear_modulus,
                                   double poisson_ratio)
    :Kernel(shear_modulus, poisson_ratio)
{
    singularity_type = "oneoverr";
    test_gradient = false;
    soln_gradient = false;
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

AdjointTractionKernel::AdjointTractionKernel(double shear_modulus,
                                   double poisson_ratio)
    :Kernel(shear_modulus, poisson_ratio)
{
    singularity_type = "oneoverr";
    test_gradient = false;
    soln_gradient = false;
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

RegularizedHypersingularKernel::RegularizedHypersingularKernel(
    double shear_modulus, double poisson_ratio)
    :Kernel(shear_modulus, poisson_ratio)
{
    singularity_type = "logr";
    test_gradient = true;
    soln_gradient = true;
    const5 = shear_modulus / (2 * PI * (1 - poisson_ratio));
}

double RegularizedHypersingularKernel::call(KernelData d, int p, int q)
{
    return const5 * ((p == q) * log(d.dist) - d.dr[p] * d.dr[q]);
}

SemiRegularizedHypersingularKernel::SemiRegularizedHypersingularKernel(
    double shear_modulus, double poisson_ratio):
    Kernel(shear_modulus, poisson_ratio),
    e{{0, 1},{-1, 0}}
{
    test_gradient = false;
    soln_gradient = true;
    singularity_type = "oneoverr";
    const double lambda =
        2 * shear_modulus * poisson_ratio / (1 - 2 * poisson_ratio);
    const1 = (1 - 2 * poisson_ratio);
    const6 = shear_modulus / (4 * PI * (1 - poisson_ratio));
    const7 = lambda / PI;
}

double SemiRegularizedHypersingularKernel::call(KernelData d, int p, int q)
{
    const double dalphadenom = 
                        (d.dist2 * d.dist * 
                         sqrt(1 - pow(d.r[0] / d.dist2, 2))); 
    const double dalphax = (d.r[0] - d.r[1]) * (d.r[0] + d.r[1]) / dalphadenom;
    const double dalphay = 2 * d.r[0] * d.r[1] / dalphadenom;
    double e_dr[]{e[0][0] * d.dr[0] + e[0][1] * d.dr[1],
         e[1][0] * d.dr[0] + e[1][1] * d.dr[1]};
    double e_n[]{e[0][0] * d.n[0] + e[0][1] * d.n[1],
         e[1][0] * d.n[0] + e[1][1] * d.n[1]};
    return 
        -(const6 / d.dist) * (
            2 * const1 * e_dr[p] * d.n[q] 
            + 2 * poisson_ratio * e_n[p] * d.dr[q]
            - 4 * e_dr[p] * d.dr[q] * d.drdn
            - (p == q) * 2 * (1 - poisson_ratio) * 
                (-d.dr[0] * d.n[1] + d.dr[1] * d.n[0])
            // I don't know why I can ignore this term. Maybe the 
            // expression given in Frangi, Novati 96 is just wrong?
            // - e[p][q] * 2 * poisson_ratio * 
            //     (-dalphax * d.n[1] + dalphay * d.n[0])
            - 2 * (1 - poisson_ratio) * e_dr[q] * d.n[p]);
}
            

HypersingularKernel::HypersingularKernel(double shear_modulus,
                                   double poisson_ratio)
    :Kernel(shear_modulus, poisson_ratio)
{
    test_gradient = false;
    soln_gradient = false;
    singularity_type = "oneoverr";
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
