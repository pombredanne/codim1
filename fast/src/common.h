/* Two macros for debugging found on http://latedev.wordpress.com/2012/08/09/c-debug-macros/
 * Added the do {} while(0) construct for safety.
 */

#define DBGVAR( os, var ) \
  (os) << "DBG: " << __FILE__ << "(" << __LINE__ << ") "\
       << #var << " = [" << (var) << "]" << std::endl

#define DBGMSG( os, msg ) \
  (os) << "DBG: " << __FILE__ << "(" << __LINE__ << ") " \
       << msg << std::endl
