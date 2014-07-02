#include <cstdlib>
#include <cmath>

/* Two macros for debugging found on http://latedev.wordpress.com/2012/08/09/c-debug-macros/
 * Added the do {} while(0) construct for safety.
 */

#define DBGVAR( outstream, var) \
  do {(outstream) << "DBG: " << __FILE__ << "(" << __LINE__ << ") "\
       << #var << " = [" << (var) << "]" << std::endl;} while(0)

#define DBGMSG( outstream, msg) \
  do {(outstream) << "DBG: " << __FILE__ << "(" << __LINE__ << ") " \
       << msg << std::endl;} while(0)

/* A function and a macro for testing comparisons of floating point values. */
inline bool AE(double a, double b, double eps = 1e-7) {
    return std::fabs(a - b) < eps; 
}

#define REQUIREAE(a, b, eps) \
    do {\
        DBGMSG(std::cout, "Assert: " << a << "  almost equal to: " << b <<\
                  "  with precision: " << eps);\
        DBGMSG(std::cout, "Difference is: " << (a - b));\
        REQUIRE(AE(a, b, eps));\
    } while(0)
