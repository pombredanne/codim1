#include "integrator.h"
#include "mapping.h"
#include "elastic_kernel.h"
#include "basis.h"

//TODO: Play with some strategy pattern ideas for making this flexible
std::vector<std::vector<double> >
double_integral(MappingEval& test_mapping_eval, 
                MappingEval& soln_mapping_eval,  
                Kernel& kernel, 
                Basis& test_basis_eval,
                Basis& soln_basis_eval,
                QuadratureInfo& test_quadrature,
                std::vector<QuadratureInfo> soln_quadrature,
                int test_basis_idx,
                int soln_basis_idx)
{
    std::vector<std::vector<double> > result(2);
    std::vector<double> result_x(2);
    std::vector<double> result_y(2);
    result[0] = result_x;
    result[1] = result_y;

    
    // Jacobian determinants are necessary to scale the integral with the
    // change of variables. 
    double test_jacobian, soln_jacobian;

    // The soln_normal is needed for the traction kernel -- the solution normal.
    // The test_normal is normally needed for the adjoint traction kernel
    // and for the hypersingular kernel -- the source normal.
    std::vector<double> test_normal, soln_normal; 

    // The basis functions are evaluated on each of the elements
    std::vector<double> test_basis_val, soln_basis_val;

    // Evaluating the kernel function requires knowledge of the physical
    // location of each quadrature point.
    std::vector<double> test_phys_pt, soln_phys_pt;

    std::vector<double> r(2);
    double kernel_val;


    test_jacobian = test_mapping_eval.get_jacobian(0.0);
    soln_jacobian = soln_mapping_eval.get_jacobian(0.0);
    test_normal = test_mapping_eval.get_normal(0.0);
    soln_normal = soln_mapping_eval.get_normal(0.0);  
    test_jacobian *= test_basis_eval.chain_rule(test_jacobian);
    soln_jacobian *= soln_basis_eval.chain_rule(soln_jacobian);

    const int num_test_pts = test_quadrature.x.size();
    for(int q_test_idx = 0; q_test_idx < num_test_pts; q_test_idx++)
    {
        const double q_pt_test = test_quadrature.x[q_test_idx];
        const double q_w_test = test_quadrature.w[q_test_idx];

        // Translate from reference segment coordinates to 
        // real, physical coordinates
        test_phys_pt = test_mapping_eval.get_physical_point(q_pt_test);

        // The basis functions should be evaluated on reference
        // coordinates
        test_basis_val = test_basis_eval.\
            evaluate_vector(test_basis_idx, q_pt_test, test_phys_pt);

        // If the integrand is singular, we need to use the appropriate
        // inner quadrature method. Which points the inner quadrature
        // chooses will depend on the current outer quadrature point
        // which will be the point of singularity, assuming same element
        const QuadratureInfo cur_soln_quad = soln_quadrature[q_test_idx];
        const int num_soln_pts = cur_soln_quad.x.size();
        for(int q_soln_idx = 0; q_soln_idx < num_soln_pts; q_soln_idx++)
        {
            const double q_pt_soln = cur_soln_quad.x[q_soln_idx];
            const double q_w_soln = cur_soln_quad.w[q_soln_idx];

            // TODO: Refactor idea: make a "get_quad_point_data" function.

            // Translate from reference segment coordinates to 
            // real, physical coordinates
            soln_phys_pt = soln_mapping_eval.get_physical_point(q_pt_soln);

            // The basis functions should be evaluated on reference
            // coordinates
            soln_basis_val = soln_basis_eval.\
                evaluate_vector(soln_basis_idx, q_pt_soln, soln_phys_pt);

            // Separation of the two quadrature points, use real,
            // physical coordinates!
            // From source to solution.
            r[0] = soln_phys_pt[0] - test_phys_pt[0];
            r[1] = soln_phys_pt[1] - test_phys_pt[1];

            // Determine the various location parameters that the kernels
            // need -- dr, drdn, drdm, dist
            KernelData kernel_data = kernel.\
                get_double_integral_data(r, test_normal, soln_normal);

            // Account for the vector form of the problem.
            // and weight by the quadrature values and the jacobian
            for (int idx_x = 0; idx_x < 2; idx_x++)
            {
                for (int idx_y = 0; idx_y < 2; idx_y++)
                {
                    // Actually perform the integration.
                    kernel_val = kernel.call(kernel_data, idx_x, idx_y);
                    result[idx_x][idx_y] += kernel_val * 
                        test_basis_val[idx_x] * soln_basis_val[idx_y] *
                        test_jacobian * soln_jacobian * q_w_test * q_w_soln;
                }
            }
        }
    }

    return result;
}

/*
 * Performs a single integral over one element. The operations are all 
 * almost identical to those in the double_integral method. Thus, read 
 * through that method for details (by some arguments, duplication of 
 * comments just creates more "code" to maintain)
 *
 * A key difference with single_integral is that the kernel function here
 * is expected to just be a standard function with a location 
 * parameter. 
 * The double integral function takes a kernel class object with 
 * a call method taking a separation input. K(x) vs. K(x - y)
 */
std::vector<std::vector<double> >
single_integral(MappingEval& soln_mapping_eval, 
                Kernel& kernel, 
                Basis& test_basis_eval,
                Basis& soln_basis_eval,
                QuadratureInfo& quadrature,
                int test_basis_idx,
                int soln_basis_idx)
{
    std::vector<std::vector<double> > result(2);
    std::vector<double> result_x(2);
    std::vector<double> result_y(2);

    result[0] = result_x;
    result[1] = result_y;

    double jacobian;
    std::vector<double> test_normal;
    jacobian = soln_mapping_eval.get_jacobian(0.0);
    test_normal = soln_mapping_eval.get_normal(0.0);
    jacobian *= test_basis_eval.chain_rule(jacobian);
    jacobian *= soln_basis_eval.chain_rule(jacobian);

    std::vector<double> phys_pt;
    std::vector<double> test_basis_val, soln_basis_val;
    double kernel_val;

    const int num_pts = quadrature.x.size();
    for(int q_idx = 0; q_idx < num_pts; q_idx++)
    {
        const double q_pt = quadrature.x[q_idx];
        const double q_w = quadrature.w[q_idx];

        phys_pt = soln_mapping_eval.get_physical_point(q_pt);

        test_basis_val = test_basis_eval.\
            evaluate_vector(test_basis_idx, q_pt, phys_pt);
        soln_basis_val = soln_basis_eval.\
            evaluate_vector(soln_basis_idx, q_pt, phys_pt);

        // Determine the various location parameters that the kernels
        // need -- dr, drdn, drdm, dist
        KernelData kernel_data =
            kernel.get_interior_integral_data(phys_pt, test_normal);

            // Account for the vector form of the problem.
            // and weight by the quadrature values and the jacobian
        for (int idx_x = 0; idx_x < 2; idx_x++)
        {
            for (int idx_y = 0; idx_y < 2; idx_y++)
            {
                // Actually perform the integration.
                kernel_val = kernel.call(kernel_data, idx_x, idx_y);
                result[idx_x][idx_y] += kernel_val * 
                    test_basis_val[idx_x] * soln_basis_val[idx_y] *
                    jacobian * q_w;
            }
        }
    }

    return result;
}

std::vector<std::vector<double> >
aligned_single_integral(MappingEval& soln_mapping_eval, 
                        Kernel& kernel, 
                        Basis& soln_basis_eval,
                        QuadratureInfo& quadrature,
                        int soln_basis_idx)
{
    std::vector<std::vector<double> > result(2, std::vector<double>(2));

    double jacobian = soln_mapping_eval.get_jacobian(0.0);
    std::vector<double> normal = soln_mapping_eval.get_normal(0.0);
    jacobian *= soln_basis_eval.chain_rule(jacobian);


    const double q_pt = quadrature.x[soln_basis_idx];
    const double q_w = quadrature.w[soln_basis_idx];

    std::vector<double> phys_pt = soln_mapping_eval.get_physical_point(q_pt);

    std::vector<double> soln_basis_val =
        soln_basis_eval.evaluate_vector(soln_basis_idx, q_pt, phys_pt);

    // Determine the various location parameters that the kernels
    // need -- dr, drdn, drdm, dist
    KernelData kernel_data =
        kernel.get_interior_integral_data(phys_pt, normal);

    // Account for the vector form of the problem.
    // and weight by the quadrature values and the jacobian
    for (int idx_x = 0; idx_x < 2; idx_x++)
    {
        for (int idx_y = 0; idx_y < 2; idx_y++)
        {
            // Actually perform the integration.
            result[idx_x][idx_y] += kernel.call(kernel_data, idx_x, idx_y) * 
                                    soln_basis_val[idx_y] * jacobian * q_w;
        }
    }

    return result;
}
