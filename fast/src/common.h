/* Two macros for debugging found on http://latedev.wordpress.com/2012/08/09/c-debug-macros/
 * Added the do {} while(0) construct for safety.
 */

#define DBGVAR( outstream, var) \
  do {(outstream) << "DBG: " << __FILE__ << "(" << __LINE__ << ") "\
       << #var << " = [" << (var) << "]" << std::endl;} while(0)

#define DBGMSG( outstream, msg) \
  do {(outstream) << "DBG: " << __FILE__ << "(" << __LINE__ << ") " \
       << msg << std::endl;} while(0)
