#include "integrator.h"
#include "mesh_eval.h"
#include "elastic_kernel.h"
#include "basis_eval.h"

//TODO: Play with some strategy pattern ideas for making this flexible
std::vector<std::vector<double> >
double_integral(MeshEval& mesh_eval, 
                bool linear_mesh,
                Kernel& kernel, 
                BasisEval& k_basis_eval,
                BasisEval& l_basis_eval,
                QuadratureInfo& k_quadrature,
                std::vector<QuadratureInfo> l_quadrature,
                int k, int i, int l, int j)
{
    std::vector<std::vector<double> > result(2);
    std::vector<double> result_x(2);
    std::vector<double> result_y(2);
    result[0] = result_x;
    result[1] = result_y;

    
    // Jacobian determinants are necessary to scale the integral with the
    // change of variables. 
    double k_jacobian, l_jacobian;

    // The l_normal is needed for the traction kernel -- the solution normal.
    // The k_normal is normally needed for the adjoint traction kernel
    // and for the hypersingular kernel -- the source normal.
    std::vector<double> k_normal, l_normal; 

    // The basis functions are evaluated on each of the elements
    std::vector<double> k_basis_val, l_basis_val;

    // Evaluating the kernel function requires knowledge of the physical
    // location of each quadrature point.
    std::vector<double> k_phys_pt, l_phys_pt;

    std::vector<double> r(2);
    double kernel_val;


    // TODO: Refactor the linear_mesh stuff out into the mesh_eval object.
    // If the mesh is linear then the jacobian and normal vectors 
    // will be the
    // same at all quadrature points, so just grab them here once.
    // TODO: There could be some "MeshInfo" object that is just 
    // grabbed once...
    if (linear_mesh)
    {
        k_jacobian = mesh_eval.get_jacobian(k, 0.0);
        l_jacobian = mesh_eval.get_jacobian(l, 0.0);
        k_normal = mesh_eval.get_normal(k, 0.0);
        l_normal = mesh_eval.get_normal(l, 0.0);  
        k_jacobian *= k_basis_eval.chain_rule(k_jacobian);
        l_jacobian *= l_basis_eval.chain_rule(l_jacobian);
    }

    const int num_k_pts = k_quadrature.x.size();
    for(int q_k_idx = 0; q_k_idx < num_k_pts; q_k_idx++)
    {
        const double q_pt_k = k_quadrature.x[q_k_idx];
        const double q_w_k = k_quadrature.w[q_k_idx];

        // Translate from reference segment coordinates to 
        // real, physical coordinates
        k_phys_pt = mesh_eval.get_physical_point(k, q_pt_k);

        // If we have a nonlinear mesh, then the jacobians and normals
        // are different for each quadrature point.
        if (!linear_mesh)
        {
            k_jacobian = mesh_eval.get_jacobian(k, q_pt_k);
            k_normal = mesh_eval.get_normal(k, q_pt_k);
            k_jacobian *= k_basis_eval.chain_rule(k_jacobian);
            l_jacobian *= l_basis_eval.chain_rule(l_jacobian);
        }

        // The basis functions should be evaluated on reference
        // coordinates
        k_basis_val = k_basis_eval.evaluate_vector(k, i, q_pt_k, k_phys_pt);

        // If the integrand is singular, we need to use the appropriate
        // inner quadrature method. Which points the inner quadrature
        // chooses will depend on the current outer quadrature point
        // which will be the point of singularity, assuming same element
        const QuadratureInfo cur_l_quad = l_quadrature[q_k_idx];
        const int num_l_pts = cur_l_quad.x.size();
        for(int q_l_idx = 0; q_l_idx < num_l_pts; q_l_idx++)
        {
            const double q_pt_l = cur_l_quad.x[q_l_idx];
            const double q_w_l = cur_l_quad.w[q_l_idx];

            // TODO: Refactor idea: make a "get_quad_point_data" function.

            // Translate from reference segment coordinates to 
            // real, physical coordinates
            l_phys_pt = mesh_eval.get_physical_point(l, q_pt_l);

            if (!linear_mesh)
            {
                // If we have a nonlinear mesh, then the jacobians and normals
                // are different for each quadrature point.
                l_jacobian = mesh_eval.get_jacobian(l, q_pt_l);
                l_normal = mesh_eval.get_normal(l, q_pt_l);
                k_jacobian *= k_basis_eval.chain_rule(k_jacobian);
                l_jacobian *= l_basis_eval.chain_rule(l_jacobian);
            }

            // The basis functions should be evaluated on reference
            // coordinates
            l_basis_val = l_basis_eval.evaluate_vector(l, j,
                                                       q_pt_l, l_phys_pt);

            // Separation of the two quadrature points, use real,
            // physical coordinates!
            // From source to solution.
            r[0] = l_phys_pt[0] - k_phys_pt[0];
            r[1] = l_phys_pt[1] - k_phys_pt[1];

            // Determine the various location parameters that the kernels
            // need -- dr, drdn, drdm, dist
            KernelData kernel_data =
                kernel.get_double_integral_data(r, k_normal, l_normal);

            // Account for the vector form of the problem.
            // and weight by the quadrature values and the jacobian
            for (int idx_x = 0; idx_x < 2; idx_x++)
            {
                for (int idx_y = 0; idx_y < 2; idx_y++)
                {
                    // Actually perform the integration.
                    kernel_val = kernel.call(kernel_data, idx_x, idx_y);
                    result[idx_x][idx_y] += kernel_val * 
                        k_basis_val[idx_x] * l_basis_val[idx_y] *
                        k_jacobian * l_jacobian * q_w_k * q_w_l;
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
single_integral(MeshEval& mesh_eval, 
                bool linear_mesh,
                Kernel& kernel, 
                BasisEval& i_basis_eval,
                BasisEval& j_basis_eval,
                QuadratureInfo& quadrature,
                int k, int i, int j)
{
    std::vector<std::vector<double> > result(2);
    std::vector<double> result_x(2);
    std::vector<double> result_y(2);

    result[0] = result_x;
    result[1] = result_y;

    double jacobian;
    std::vector<double> k_normal;
    if (linear_mesh)
    {
        jacobian = mesh_eval.get_jacobian(k, 0.0);
        k_normal = mesh_eval.get_normal(k, 0.0);
        jacobian *= i_basis_eval.chain_rule(jacobian);
        jacobian *= j_basis_eval.chain_rule(jacobian);
    }

    std::vector<double> phys_pt;
    std::vector<double> i_basis_val, j_basis_val;
    std::vector<double> r(2);
    double kernel_val;

    const int num_pts = quadrature.x.size();
    for(int q_idx = 0; q_idx < num_pts; q_idx++)
    {
        const double q_pt = quadrature.x[q_idx];
        const double q_w = quadrature.w[q_idx];

        phys_pt = mesh_eval.get_physical_point(k, q_pt);

        if (!linear_mesh)
        {
            jacobian = mesh_eval.get_jacobian(k, q_pt);
            k_normal = mesh_eval.get_normal(k, q_pt);
            jacobian *= i_basis_eval.chain_rule(jacobian);
            jacobian *= j_basis_eval.chain_rule(jacobian);
        }

        i_basis_val = i_basis_eval.evaluate_vector(k, i, q_pt, phys_pt);
        j_basis_val = j_basis_eval.evaluate_vector(k, j, q_pt, phys_pt);

        // Determine the various location parameters that the kernels
        // need -- dr, drdn, drdm, dist
        KernelData kernel_data =
            kernel.get_interior_integral_data(phys_pt, k_normal);

            // Account for the vector form of the problem.
            // and weight by the quadrature values and the jacobian
        for (int idx_x = 0; idx_x < 2; idx_x++)
        {
            for (int idx_y = 0; idx_y < 2; idx_y++)
            {
                // Actually perform the integration.
                kernel_val = kernel.call(kernel_data, idx_x, idx_y);
                result[idx_x][idx_y] += kernel_val * 
                    i_basis_val[idx_x] * j_basis_val[idx_y] *
                    jacobian * q_w;
            }
        }
    }

    return result;
}
