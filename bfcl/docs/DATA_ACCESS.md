# Dataset Access Contract

The setup phase may read the four pinned BFCL v4 multi-turn source files and
compile public function schemas to create a hashed manifest. It must not execute
cases.

After UTF-8 sorting and one `Random(0)` shuffle, the first 40 families are sealed
test, the next 40 are dev, and the remaining 120 are train. Train evolution must
open one recorded 40-family round. Dev opens only for paired Skill A/B
validation and never creates an update. Ordinary code cannot open test. Final
test requires a frozen plan hash, manifest hash, owner identity, and exclusive
durable marker. An active marker can resume only for the identical owner and
plan. A completed marker can only verify and reload its bound immutable report;
it cannot authorize another evaluation.

The final runner writes a plan-bound dispatch claim before each unit, reconciles
an unknown in-flight unit through the executor recovery seam, persists every
official result/trajectory/accounting/latency artifact atomically, and resumes
only a contiguous manifest prefix. This keeps a crash from consuming authority
while losing results or rerunning an externally completed case.

Latency is measured by the runner, not supplied by the executor. Each complete
Vanilla, G000, and G003 run has a hash-verified monotonic start, wall-clock
start, boot ID, durable completion checkpoint, and authoritative receipt.
Recovery uses monotonic time on the same boot and the wall-clock anchor across
boots. The run closes only after its last evaluator result and durable run
completion are fsynced. Reports use those direct system-run durations, never a
sum of unit durations; unit timing is retained only as diagnostics.

BFCL user turns are fixed dataset content. There is no User Simulator model or
User Simulator token usage in this adaptation. Possible answers, reference
calls, checker details, and raw exception content cannot enter online prompts or
offline updates. Updaters receive only official `valid`, a sanitized error
class, and host-defined visible execution-failure classes.
