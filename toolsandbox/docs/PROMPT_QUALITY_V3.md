# Online prompt quality v3

The user explicitly authorized Policy, Critic, and Revision prompt updates after the fixed train smoke exposed optional-argument false positives, unsupported time values, and incorrect Skill bindings.

The three prompts retain strict structured output, the current augmented tool schema and visibility boundary, deterministic Controller decisions, and at most one Revision. Policy separates task language from actual query criteria, obtains missing values through available tools, and permits null Skill binding. Critic distinguishes schema-required fields from optional filters and review triggers from blocking errors. Revision validates every final argument and rebinds or clears the Skill when changing tools.

Only prompt text and its version acceptance changed for this comparison. Model weights, decoding, output ceilings, retrieval settings, native scenarios, and evaluator are unchanged. The prompt manifest records v3 for all three roles; old v1 requests and Critic v2 requests remain readable. Existing filenames are retained by the fixed loader allowlist; manifest version and content hash identify the revision.

## Validation

373 provider, online, and orchestration tests passed. Six local GPU checks on previously captured train inputs returned valid, untruncated outputs; the checks include an accepted action, grounded-field rejection, and Critic-to-Revision Skill mismatch handling. These probes do not provide native episode scores or prove generalization.

The fixed comparison uses the same two train scenarios, followed by Policy-memory reflection and G001 retesting. Its baseline is reflection-smoke-20260912T100014-593e59, with native similarities 0.5, 0.0, 0.5, 0.5, two NONE reflections, and no truncation. The v3 run is reflection-smoke-20260912T101456-604468. Runtime artifacts, baseline prompt bytes, and candidate rationale are under /root/toolsandbox-runtime/proposed-fixes/prompt-quality-v3/.

This is development evidence. Formal token calibration must be rerun for the new prompt hashes; provisional smoke limits must not be relabeled as calibrated. Remote user responses also vary, so this single train comparison cannot isolate a causal effect or establish test performance.

## Follow-up selection

v4 was tested in reflection-smoke-20260912T102630-b6e19f and rejected: scores 0.5, 0.0, 0.5, followed by a Policy length termination before the fourth evaluation. Exact v3 prompt bytes and manifest were restored. The full comparison is /root/toolsandbox-runtime/prompt-quality-comparison.md. No formal training has started.
