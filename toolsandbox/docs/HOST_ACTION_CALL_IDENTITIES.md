# Host-assigned action call identity

The user explicitly authorized assigning call IDs by position while retaining
model output for audit after a real Policy batch reused one label twice. This
approved wire-to-host decoding exception narrows the earlier batch-ID rule in
`pipeline.md`: model labels may repeat; the public `ActionEnvelope` still requires
unique IDs. Prompts, public output schema, tools and evaluation are unchanged.

Three identities remain distinguishable:

1. `model_call_id`: the model's original nonempty string, retained verbatim in the
   original provider response blob. Repeated labels do not merge calls.
2. `host_action_call_id`: decoder version `host-action-call-identity-v1` hashes the
   logical request ID, state/unit reference, input fingerprint, role and zero-based
   position. Physical retry IDs are excluded. All non-ID fields and order survive
   unchanged. The private wire reader relaxes only batch label uniqueness; all
   other structure, types, required fields, lengths and strict JSON rules remain.
3. Native `exec_…` ID: the existing dispatch adapter derives this from the committed
   turn, final action and slot. `execution_call_ids(decision)` reconstructs it.
   Correspondence to the final model request is positional, including after Revision.

The online response seam writes an `action_call_identity` checkpoint containing
version, logical request, source attempt, original raw response SHA256, decoded
ActionEnvelope SHA256 and ordered raw-label/host-label mappings. The existing
ledger keeps the unmodified raw bytes and separately stores the validated action.
An audit checkpoint alone does not mark a response complete; the normal durable
response commit and application steps still govern execution.

Restoring a normalized response recomputes and verifies its mapping against the
stored original response. Older completed responses whose original strict action
already matches their stored value restore unchanged, without rewriting IDs or
adding a normalization claim. A failed old run is not resumed under the new decoder;
new real experiments use a new run directory and source manifest identity.

Offline revalidation of the failed Policy response with raw SHA256
`46dd5661a6ec402472e9c3f832a6f67c79a803a4f6ffeddd9330d30063586dc3`
produced two distinct host IDs from two equal model labels. Both calls, all
arguments and their order were retained; the original raw file and SHA256 remained
unchanged. No model or external API call was made for this verification.
