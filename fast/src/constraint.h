#ifndef __codim1_constraint_h
#define __codim1_constraint_h

#include "linalg.h"
#include <utility>
#include <vector>
#include <unordered_map>

/* A list of matrix constraints is generated from the mesh
 * connectivity and boundary conditions. These constraints
 * are represented by an integer referring to the relevant
 * degree of freedom and a double that multiplies that 
 * dofs value in the linear system. A rhs value is represented
 * by a dof id of -1. 
 * So, for example:
 * 4x + 3y = 2 ====> 4x + 3y - 2 = 0
 * would be a constraint list consisting of the tuples
 * [(x_dof, 4), (y_dof, 3), (RHS, -2)]
 *
 * Constraints are used to ensure continuity between
 * elements and to enforce displacement discontinuities between elements
 * that connect to another element with a displacement discontinuity 
 * boundary condition or solution type.
 *
 * Continuity constraints for a Lagrange interpolating basis that has a
 * node at the boundary will simply be of the form [(e1_dof, 1), (e2_dof, -1)]
 * because the two boundary dofs are set equal.
 *
 * TODO: CHANGE TO USE BOOST_STRONG_TYPEDEF?
 */ 
namespace codim1
{
    extern const int RHS = -1;

    typedef std::pair<int, double> DOFWeight;

    /* A constraint is stored as a list of DOFs and weights, where the
     * first dof is implicitly the constrained dof.
     */
    typedef std::vector<DOFWeight> Constraint;

    /* I store a constraint matrix as a map from the constrained dof to
     * the constraint on that dof.
     */
    typedef std::unordered_map<int, Constraint> ConstraintMatrix;

    /* Constrain two degrees of freedom to be identical. */
    Constraint continuity_constraint(int dof1, int dof2);

    /* This creates a constraint representing the offset between the value
     * of two DOFs. Primarily useful for imposing a fault slip as the 
     * displacement jump when a fault intersects the surface.
     * The direction of offset is from dof1 to dof2. So,
     * value[dof1] + offset = value[dof2]
     */
    Constraint offset_constraint(int dof1, int dof2, double offset);

    void add_constraint(ConstraintMatrix& cm, Constraint c);

    void add_mat_with_constraints(MatrixEntry entry,
                                  mat& mat, vec& rhs,
                                  ConstraintMatrix cm);

    void add_rhs_with_constraints(DOFWeight entry, 
                                  vec& rhs,
                                  ConstraintMatrix cm);
}

#endif
