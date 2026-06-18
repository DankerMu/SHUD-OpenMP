/* test_writer_main.cpp — minimal C++ harness for compose_snapshot_filename.
 *
 * Companion to tools/rhs_snapshot/test_writer.sh (F9 fix; PR #54 round-1).
 * Invoked as:
 *
 *   test_writer <t_value> <suffix>
 *
 * Behavior (matches the API at tools/rhs_snapshot/writer.h):
 *   - Empty string returned by compose_snapshot_filename signals rejection
 *     (path-traversal `/` `\\` or F5 64-char length cap). On rejection the
 *     harness prints nothing and exits non-zero (1) so the shell test
 *     can `assert_exit 1`.
 *   - Non-empty return string is printed to stdout (no trailing newline
 *     beyond the single \n appended below) and the harness exits 0.
 *
 * Why this exists: F9 wants 4-scenario coverage (empty suffix / non-empty
 * suffix / `/` rejection / `\\` rejection) of the host-side compose
 * helper, paralleling the cvode_stats_diff test suite's pattern.
 *
 * Build: see tools/rhs_snapshot/test_writer.sh, which g++-compiles this
 * file + writer.cpp into a temp binary and runs the scenarios.
 */
#include "writer.h"
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <t_value> <suffix>\n", argv[0]);
        return 2;
    }
    const double      t      = std::strtod(argv[1], nullptr);
    const std::string suffix = argv[2];
    const std::string out    = shud_snap::compose_snapshot_filename(t, suffix);
    if (out.empty()) {
        return 1;  /* rejected: path-traversal or F5 length cap */
    }
    std::printf("%s\n", out.c_str());
    return 0;
}
