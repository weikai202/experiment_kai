# Failure-Mode Attribution and Stability Analysis

Status: `approved`

## Purpose

This document fixes the post-evaluation analysis used to answer two questions:

1. Which ToolSandbox test variants are directly related to failure modes observed
   and retained by the evolution pipeline during training?
2. How many of those variants, and how many complete eight-variant families, are
   repaired by Updated relative to Generation-0?

This is a mechanism-linked observational analysis. It does not establish causal
effect and must not use an LLM to assign relationships after results are known.

## Frozen Match Signature

Task 016 creates each signature before final evaluation from exactly:

```text
skill_id
evidence_kind
sorted_public_canonical_tool_dependencies
sanitized_outcome_class
```

Allowed `evidence_kind` values are the finite host-defined classes for a visible
tool exception, fixture miss, controller rejection, selected related milestone
shortfall, minefield hit, or visible terminal-state failure. Outcome classes are
finite public error/controller categories, never raw messages. Canonical JSON and
the project SHA-256 function determine `failure_signature_sha256`.

A lineage record links that signature to its training `mode_id`, producing round,
source evidence hash, committed effect, and any accepted evolved Skill version.
It cannot contain scenario text, concrete entities, raw exceptions, tool payloads,
hidden evaluator content, prompts, model output, or endpoint information.

## Test-Time Matching

Task 018 derives the same signature fields from a completed trusted trajectory
only after all Vanilla, Generation-0, and Updated runs are durably closed. Matching
is exact equality. There is no text similarity, embedding, fuzzy rule, manual
override, or model judgment.

Classify a Generation-0 failure as:

- `unmatched`: zero lineage matches;
- `ambiguous`: more than one distinct lineage match;
- `related_unrepaired`: one match but Updated is not fully successful, or the
  linked evolved Skill version is not proven used by Updated;
- `related_repaired`: one match, Generation-0 is not fully successful, Updated is
  fully successful, and Updated's committed decision-chain audit proves use of
  the linked evolved Skill version.

Incomplete inputs receive `incomplete_evidence` and never enter repaired counts.
Store IDs and proof hashes, not restricted trajectory content.

## Scenario Metrics

For paired scenarios report:

```text
related_case_count
repaired_case_count
repair_rate = repaired_case_count / related_case_count
unmatched_case_count
ambiguous_case_count
incomplete_evidence_case_count
mean_similarity(updated) - mean_similarity(generation_0), overall
mean paired similarity difference, related cases only
secondary Updated-minus-Vanilla values
```

Use `math.fsum` in manifest scenario order. A zero related-case denominator yields
`repair_rate: null` with an explicit completeness flag, not zero.

## Family Stability Metrics

ToolSandbox groups exactly eight perturbation variants under each
`scenario_family_id`. For every system and family compute:

```text
all_variants_fully_successful
minimum_native_similarity
similarity_range = maximum - minimum
related_generation_0_failure_variant_count
repaired_variant_count
all_related_failures_repaired
```

`all_related_failures_repaired` is defined only when the related failure count is
positive and every such variant is `related_repaired`. Aggregate the number/rate
of all-variant-success families, mean family minimum, mean family range, related
families, and fully repaired related families. Confidence intervals continue to
sample family IDs, never individual variants.

## Publication Rules

Publish only content-free attribution rows, aggregate summaries, manifest hashes,
and the statement that attribution is observational rather than causal. Preserve
unmatched, ambiguous, and incomplete counts. Never hand-edit a classification,
drop a failed variant, or rerun one system based on this analysis.
