Verifier verdict for: F2 — CI LEAK SUMMARY gate broken
Reviewed head SHA: 09a815dbcab9eabbddcad9550c706ddfa8636519
Verdict: CONFIRMED
Evidence:
- Defect (a) wrong-backend regex CONFIRMED: .github/workflows/serial-baseline.yml:1449 regex
  `^==[0-9]+==(ERROR: LeakSanitizer:|SUMMARY: AddressSanitizer: detected memory leaks)|LEAK SUMMARY:`
  matches LSan-LEAK-DETECTED stderr or valgrind output. The step comment at lines 1415-1419
  ("ASan exit path emits a `LEAK SUMMARY:` line on stderr, evidence that the C++ destructor
  chain ran to completion") is factually wrong: ASan/LSan never prints `LEAK SUMMARY:` (that
  string is valgrind memcheck), and the ERROR/SUMMARY-LeakSanitizer lines are emitted ONLY
  when leaks are detected. On a clean dtor-full run with detect_leaks=1 (line 1427 env),
  LSan stderr is silent w.r.t. all three alternatives → leak_summary_count=0 → exit 1 at
  line 1466 with "ASan dtor coverage missing". Gate semantics are inverted: the GOOD case
  hard-fails.
- Defect (b) unreachable on leak case CONFIRMED: line 1441 asan_err grep pattern
  `AddressSanitizer.*ERROR|==.*==ERROR` matches the same `==<pid>==ERROR: LeakSanitizer:`
  line that the leak_summary_count grep at 1449 targets. The `if [[ "$asan_err" != "0" ... ]]`
  check at line 1458 with `exit 1` at line 1461 runs BEFORE the leak_summary_count check at
  line 1463. So on a real-leak path the leak_summary_count gate is dead code (asan_err>=1
  short-circuits first); on a clean path the gate fires (defect a). Either way the gate is
  broken.
- Defect (c) skipped on PR-400 CONFIRMED: line 1413 step `if:` requires
  `steps.data_probe.outputs.data_available == 'true'`. CI log for job 83998346479 at
  2026-06-29T07:29:12.9518805Z emits `::notice CI runner has no keliya forcing/input data;
  build verification PASSED, sanitizer run SKIPPED.`, proving data_probe returned
  data_available=false on the GH-hosted runner (no keliya forcing/X*.csv at
  SHUD/Basins/keliya/forcing/). The ASan run step did not execute, no grep was performed,
  and the broken gate was not exercised on PR-400 head 09a815d. PR claim that the gate
  "exercises new LEAK SUMMARY: grep gate" on this head is contradicted by the log.
Note: All three sub-defects independently confirmed from the workflow file + CI log; even one would suffice to invalidate the claimed gate. Original `a7eb922` _exit(0) rationale (Model_Data dtor uninit-pointer at large NumEle) is consistent with the spec's stated motivation for dtor-coverage but is the very regression mode the gate is supposed to catch — and won't.
