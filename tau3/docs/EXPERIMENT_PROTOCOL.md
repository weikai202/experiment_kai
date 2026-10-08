# tau3 Evolution Experiment Protocol

## Benchmark boundary

The source benchmark is tau3/tau2-bench commit
`17e07b1da2bbc0cadfddeea36412686e0604127b`. Formal experiments include only the
Airline, Retail, and Telecom text domains. The native environment, tools, user simulator, task
semantics, and evaluator remain authoritative. The adapter supplies an Assistant participant and
does not execute tools itself.

The tau2 package at the pinned source currently declares Python 3.12 or newer. This project remains
a standalone Python 3.10 artifact by keeping native objects behind a lazy adapter seam. The live
benchmark executor must run in a separately reviewed pinned tau runtime; that runtime is not needed
for offline contract tests.

## Pipeline-aligned split

The official 100-task test set is preserved byte-for-byte in official order and stored in a sealed
manifest. Only the official 178 training tasks are partitioned. For each domain independently, IDs
are UTF-8 sorted and shuffled once with Python 3.10 `random.Random(0)`:

| Domain | Evolution train | Dev | Official test | Per round |
|---|---:|---:|---:|---:|
| Airline | 24 | 6 | 20 | 8 |
| Retail | 60 | 14 | 40 | 20 |
| Telecom | 60 | 14 | 40 | 20 |
| Total | 144 | 34 | 100 | 48 |

The first 24/60/60 shuffled tasks are split into three equal, contiguous domain shards. The remainder
is Dev. A task is the clustering unit. Multiple simulator seeds or trials for one task are execution
variants and must follow that task; they are not pseudo-families and cannot cross splits.

`official178_no_dev_v1` retains all 178 official train tasks for compatibility with earlier
baselines. It is a different experimental protocol. Metrics may be compared only when protocol,
manifest hash, official-source hash, and sealed-test hash all match.

## Evolution and Dev

Round 0 evolves G000 to G001, Round 1 evolves G001 to G002, and Round 2 evolves G002 to G003.
Only the assigned 48 train tasks can create failure modes, Memory changes, or Skill candidates.


G000 is compiled deterministically from the pinned public Airline, Retail, and Telecom Agent tool
schemas. Task prompts, evaluator material, private state, and user tools are forbidden inputs. The
source commit, source-file hashes, schema hashes, and library hash bind every generation. Formal
Qwen calls require a real train-only smoke receipt. The limit is the observed maximum plus
`max(64, ceil(20%))`, rounded to 64; a length-truncated sample invalidates the receipt.
Dev can only accept or reject an existing Skill candidate. Selection first filters by declared
domain and native tool dependencies, then orders eligible tasks by SHA-256 over seed, Skill ID,
domain, and task ID. At most 20 tasks are selected. Previous and candidate branches share the task,
simulator setup, model configuration, starting generation, and all artifacts except the evaluated
Skill. A candidate is accepted if it increases native full-success count, or if that count ties and
the sum of native rewards increases. Incomplete branches and empty relevant sets are rejected.

Each final task-trial is claimed before dispatch. Unknown outcomes are reconciled through the
executor recovery seam and are never blindly re-dispatched. The runner validates the persisted
native trusted trace, authoritative accounting ledger, execution binding, and durable unit
material.

## Final evaluation

The sealed official 100-task test is executed once in fixed order for Vanilla, G000, and fixed G003.
G003 is not selected using test results. Diagnostic best checkpoints may be retained but are not a
formal system. Every system uses the same task clusters and simulator seeds.

Primary outputs are native mean reward, full-success rate, per-domain results, direct latency, all
reported tokens, and effective Qwen output-token cost. Failure repair is observational: a case is

Same-boot latency uses a persisted monotonic start and closes after durable material persistence.
Cross-boot recovery uses the persisted wall-clock material marker and labels that clock explicitly.
related only by one exact host-derived observable signature, and repaired only when G000 fails,
G003 succeeds, and the trusted G003 trajectory proves use of the linked evolved Skill version.

## Accounting

Direct round latency begins immediately before live round execution and ends after durable material
checkpoint persistence. All physical Qwen, embedding, and user-simulator attempts contribute to
token totals. Missing provider usage keeps totals incomplete. `total_cost` is non-monetary and sums
only Qwen output tokens whose exact response is causally bound to a committed substantive Memory or
Skill effect. Rejected, NONE, SKIP, and no-op responses contribute zero.
