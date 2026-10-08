# Failure-Mode Attribution

Before final evaluation, each committed train failure lineage freezes an exact
signature over Skill ID, one host-defined evidence kind, sorted public canonical
tool dependencies, and a sanitized outcome class. Raw prompts, entities, tool
payloads, exceptions, ground truth, and evaluator internals are excluded.

After all three final systems close, G000 failure signatures are derived from
trusted executed trajectories. Exact equality produces one of: `unmatched`,
`ambiguous`, `related_unrepaired`, or `related_repaired`. Missing evidence is
`incomplete_evidence`. A related repair additionally requires official G003
success and host-derived proof that G003 used the lineage's accepted Skill
version. That proof is a hash-verified chain from a generation-scoped retrieval
receipt through the exact final decision and accepted online response
application to an executed call and its substantive committed effect. Case,
turn, and generation are part of call identity. Caller-authored proof tuples,
uncommitted calls, no-op effects, and effects for another call are rejected.
Final reporting loads that proof only from the hash-verified, plan-ordered unit
artifact produced by the runner; it rejects duplicate, out-of-plan, or
caller-constructed result rows.
For G000 and G003, the dispatch, artifact, and trusted trajectory must also
match the plan's exact generation and resource-manifest hashes. A different
self-consistent generation with the same textual ID is rejected.
When a native execution receipt substitutes for a full trajectory, its exact
generation/resource hashes and official-result hash are enforced here. The
authenticity of its native state digest remains the responsibility of the
reviewed concrete BFCL execution adapter that issues the receipt.

Report related cases, repaired cases, repair rate, and every excluded class.
For each family and system, report all-four-variants success, valid variant
count, related G000 failures, repaired variants, and whether every related
failure was repaired. Never use fuzzy matching, embeddings, an LLM, or manual
overrides after results are known. These are mechanism-linked observational
associations, not causal effects.
